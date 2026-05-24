import { Capacitor } from '@capacitor/core';
import { Camera as CapCamera } from '@capacitor/camera';
import { extract_ml_features, SparseRandomForest, DtwEngine } from './engine.js';

export class CameraEngine {
    constructor() {
        this.videoElement = document.getElementById('webcam');
        this.canvasElement = document.getElementById('output_canvas');
        this.canvasCtx = this.canvasElement.getContext('2d');
        
        this.btnStart = document.getElementById('btn-start');
        this.startText = document.getElementById('start-text');
        this.translationText = document.getElementById('translation-text');
        
        this.isRecording = false;
        this.sequenceBuffer = [];
        this.templates = [];
        this.staticModel = null;
        this.sentenceWords = [];
        this.wordCandidates = []; // Top-3 candidates per recognized word, for history
        this.history = JSON.parse(localStorage.getItem('signHistory') || '[]');

        this.selectedSourceLang = localStorage.getItem('selectedSourceLang') || 'ASL';
        this.selectedTargetLang = localStorage.getItem('selectedTargetLang') || 'English';
        this.signLanguages = ['ASL', 'TSL', 'CSL', 'BSL'];
        this.spokenLanguages = ['English', 'Thai', 'Chinese', 'Spanish', 'Japanese'];
        this.currentModalTab = 'sign';
        this.modalContext = 'source';
        this.tempSelectedLang = '';
        
        this.btnClear = document.getElementById('btn-clear');
        this.btnCopy = document.getElementById('btn-copy');
        this.btnSpeak = document.getElementById('btn-speak');
        
        this.toggleHolistic = document.getElementById('toggle-holistic');
        this.toggleHolisticDot = document.getElementById('toggle-holistic-dot');
        this.showSkeleton = true;
        
        // Create an offscreen canvas to pre-flip the video frame horizontally before sending
        // it to MediaPipe Holistic, matching Python's cv2.flip(image, 1) coordinate alignment.
        this.offscreenCanvas = document.createElement('canvas');
        this.offscreenCanvas.width = 320;
        this.offscreenCanvas.height = 240;
        this.offscreenCtx = this.offscreenCanvas.getContext('2d');

        // Setup initial canvas mirroring to be the opposite of video mirroring.
        // Since the coordinates from MediaPipe will be in the flipped space,
        // drawing directly on an unmirrored canvas will align with a mirrored video.
        if (this.videoElement.classList.contains('-scale-x-100')) {
            this.canvasElement.classList.remove('-scale-x-100');
        } else {
            this.canvasElement.classList.add('-scale-x-100');
        }
        
        // Initialize MediaPipe
        this.holistic = new window.Holistic({locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
        }});
        
        const savedComplexity = parseInt(localStorage.getItem('modelComplexity') || '2', 10);
        this.holistic.setOptions({
            modelComplexity: savedComplexity,  // Checked dynamically from settings (1 = Low, 2 = High)
            smoothLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        this.holistic.onResults(this.onResults.bind(this));
        
        this.camera = new window.Camera(this.videoElement, {
            onFrame: async () => {
                // Pre-flip video frame horizontally onto the offscreen canvas
                this.offscreenCtx.save();
                this.offscreenCtx.translate(320, 0);
                this.offscreenCtx.scale(-1, 1);
                this.offscreenCtx.drawImage(this.videoElement, 0, 0, 320, 240);
                this.offscreenCtx.restore();

                await this.holistic.send({image: this.offscreenCanvas});
            },
            width: 320,
            height: 240
        });
    }

    async initialize() {
        this.translationText.innerText = "Loading AI Models...";
        
        // Update language button text on load
        const btnSource = document.getElementById('btn-lang-source');
        const btnTarget = document.getElementById('btn-lang-target');
        if (btnSource) btnSource.innerText = this.selectedSourceLang;
        if (btnTarget) btnTarget.innerText = this.selectedTargetLang;
        
        try {
            // Load Models
            const resStatic = await fetch('/model_output/static_forest.json');
            this.staticModel = new SparseRandomForest(await resStatic.json());
            
            const resDynamic = await fetch('/model_output/dynamic_forest.json');
            this.dynamicModel = new SparseRandomForest(await resDynamic.json());

            const resCluster = await fetch('/model_output/cluster_model.json');
            this.clusterModel = await resCluster.json();

            const resClusterMap = await fetch('/model_output/file_cluster_mapping.json');
            const clusterMapRaw = await resClusterMap.json();
            this.fileClusterMapping = {};
            for (const key in clusterMapRaw) {
                const cleanKey = key.replace('.json', '');
                this.fileClusterMapping[cleanKey] = clusterMapRaw[key];
            }
            
            // Load Templates
            const resMan = await fetch('/templates_manifest.json');
            const manifest = await resMan.json();
            
            for (const path of manifest) {
                const templatesIdx = path.indexOf('templates/');
                const cleanPath = templatesIdx !== -1 ? '/' + path.substring(templatesIdx) : path;
                const resTpl = await fetch(cleanPath);
                const seq = await resTpl.json();
                
                // Pre-compute DTW features (120 dimensions to match C++)
                const features = seq.map(frame => {
                    const feat = extract_ml_features(frame);
                    return feat ? feat.slice(0, 120) : null;
                }).filter(f => f !== null);

                if (features.length > 0) {
                    const label = path.split('/').pop().replace('.json', '');
                    
                    // Extract category (e.g. movement/single_hand vs movement/2_hands)
                    let category = "";
                    const normalizedPath = path.replace(/\\/g, '/');
                    const parts = normalizedPath.split('/');
                    const tplIndex = parts.indexOf('templates');
                    if (tplIndex !== -1 && tplIndex + 2 < parts.length) {
                        category = parts.slice(tplIndex + 1, parts.length - 1).join('/');
                    }
                    this.templates.push({ label, features, category });
                }
            }
            console.log(`Loaded ${this.templates.length} templates.`);
            this.translationText.innerText = "Ready to Translate";
            
        } catch(e) {
            console.error("Failed to load models:", e);
            this.translationText.innerText = "Error Loading Engine";
        }
        if (Capacitor.isNativePlatform()) {
            try {
                const permission = await CapCamera.requestPermissions();
                if (permission.camera !== 'granted' && permission.camera !== 'granted-system') {
                    this.translationText.innerText = "Camera Permission Required";
                    return;
                }
            } catch (e) {
                console.warn("Camera permission error:", e);
            }
        }

        // Action Buttons Listeners
        this.btnClear.addEventListener('click', () => {
            this.sentenceWords = [];
            this.wordCandidates = [];
            this.translationText.innerText = "Waiting for sign...";
            this.updateTranslateButton();
        });

        this.btnCopy.addEventListener('click', () => {
            if (this.sentenceWords.length > 0) {
                navigator.clipboard.writeText(this.sentenceWords.join(' '));
                const origText = this.translationText.innerText;
                this.translationText.innerText = "Copied to clipboard!";
                setTimeout(() => {
                    this.translationText.innerText = origText;
                }, 1000);
            }
        });

        this.btnSpeak.addEventListener('click', () => {
            if (this.sentenceWords.length > 0) {
                const speech = new SpeechSynthesisUtterance(this.sentenceWords.join(' '));
                window.speechSynthesis.speak(speech);
            }
        });

        if (this.toggleHolistic && this.toggleHolisticDot) {
            this.toggleHolistic.addEventListener('click', () => {
                this.showSkeleton = !this.showSkeleton;
                if (this.showSkeleton) {
                    this.toggleHolistic.classList.replace('bg-slate-200', 'bg-brand-blue');
                    this.toggleHolisticDot.classList.replace('translate-x-0', 'translate-x-5');
                } else {
                    this.toggleHolistic.classList.replace('bg-brand-blue', 'bg-slate-200');
                    this.toggleHolisticDot.classList.replace('translate-x-5', 'translate-x-0');
                }
            });
        }

        // Camera Flip/Mirror Toggle — only toggles the display mirror on the video element.
        // The actual MediaPipe input is always pre-flipped via offscreenCanvas to match Python.
        // The landmark overlay canvas is NOT flipped since coords are in the correct space.
        const btnFlip = document.getElementById('btn-camera-flip');
        if (btnFlip) {
            btnFlip.addEventListener('click', () => {
                this.videoElement.classList.toggle('-scale-x-100');
                this.canvasElement.classList.toggle('-scale-x-100');
            });
        }

        // Debug Panel Trigger
        const appTitle = document.getElementById('app-title');
        const debugPanel = document.getElementById('debug-panel');
        const btnCloseDebug = document.getElementById('btn-close-debug');
        if (appTitle && debugPanel) {
            appTitle.addEventListener('click', () => {
                debugPanel.classList.remove('hidden');
            });
        }
        if (btnCloseDebug && debugPanel) {
            btnCloseDebug.addEventListener('click', () => {
                debugPanel.classList.add('hidden');
            });
        }

        // Translate to Sentence Button (Gemini API)
        const btnAiTranslate = document.getElementById('btn-ai-translate');
        if (btnAiTranslate) {
            btnAiTranslate.addEventListener('click', () => this.translateSentence());
        }

        // History: clear all
        const btnClearHistory = document.getElementById('btn-clear-history');
        if (btnClearHistory) {
            btnClearHistory.addEventListener('click', () => {
                if (confirm('Clear all history?')) {
                    this.history = [];
                    localStorage.removeItem('signHistory');
                    this.renderHistory();
                }
            });
        }

        // History: back button
        const btnBackHistory = document.getElementById('btn-back-history');
        if (btnBackHistory) {
            btnBackHistory.addEventListener('click', () => {
                const panel = document.getElementById('history-detail');
                if (panel) panel.classList.add('translate-y-full');
            });
        }

        // Settings: Gemini API key persistence
        const apiKeyInput = document.getElementById('gemini-api-key');
        if (apiKeyInput) {
            apiKeyInput.value = localStorage.getItem('geminiApiKey') || '';
            apiKeyInput.addEventListener('input', () => {
                localStorage.setItem('geminiApiKey', apiKeyInput.value.trim());
                localStorage.removeItem('geminiActiveModel');
            });
        }

        // Settings: Theme Toggle (Light / Dark Mode with smooth fade transition)
        const themeBtn = document.getElementById('btn-theme-toggle');
        const themeDot = document.getElementById('btn-theme-dot');
        const themeStatus = document.getElementById('theme-status');
        const themeIcon = document.getElementById('theme-icon');

        const setThemeMode = (isDark) => {
            if (isDark) {
                document.body.classList.add('dark');
                if (themeBtn) themeBtn.classList.replace('bg-slate-200', 'bg-brand-blue');
                if (themeDot) themeDot.classList.replace('translate-x-0', 'translate-x-5');
                if (themeStatus) themeStatus.innerText = "Dark mode active";
                if (themeIcon) {
                    themeIcon.className = "ph ph-moon text-xl text-brand-blue mr-3";
                }
                localStorage.setItem('themeMode', 'dark');
            } else {
                document.body.classList.remove('dark');
                if (themeBtn) themeBtn.classList.replace('bg-brand-blue', 'bg-slate-200');
                if (themeDot) themeDot.classList.replace('translate-x-5', 'translate-x-0');
                if (themeStatus) themeStatus.innerText = "Light mode active";
                if (themeIcon) {
                    themeIcon.className = "ph ph-sun text-xl text-slate-400 mr-3";
                }
                localStorage.setItem('themeMode', 'light');
            }
        };

        // Load saved theme on boot
        const currentTheme = localStorage.getItem('themeMode') || 'light';
        setThemeMode(currentTheme === 'dark');

        if (themeBtn) {
            themeBtn.addEventListener('click', () => {
                const isCurrentlyDark = document.body.classList.contains('dark');
                setThemeMode(!isCurrentlyDark);
            });
        }

        // Settings: Model Complexity Toggle
        const btnLow = document.getElementById('btn-complexity-low');
        const btnHigh = document.getElementById('btn-complexity-high');

        const setComplexityUI = (complexity) => {
            if (complexity === 1) {
                if (btnLow) {
                    btnLow.className = "px-3 py-1 text-xs font-semibold bg-brand-blue text-white rounded-md shadow-sm transition-all";
                }
                if (btnHigh) {
                    btnHigh.className = "px-3 py-1 text-xs font-semibold text-slate-500 hover:text-slate-700 dark:text-slate-400 transition-all";
                }
            } else {
                if (btnLow) {
                    btnLow.className = "px-3 py-1 text-xs font-semibold text-slate-500 hover:text-slate-700 dark:text-slate-400 transition-all";
                }
                if (btnHigh) {
                    btnHigh.className = "px-3 py-1 text-xs font-semibold bg-brand-blue text-white rounded-md shadow-sm transition-all";
                }
            }
        };

        // Initialize complexity UI state
        const initialComplexity = parseInt(localStorage.getItem('modelComplexity') || '2', 10);
        setComplexityUI(initialComplexity);

        if (btnLow) {
            btnLow.addEventListener('click', () => {
                localStorage.setItem('modelComplexity', '1');
                setComplexityUI(1);
                this.holistic.setOptions({ modelComplexity: 1 });
                console.log("[AI] MediaPipe Holistic Model Complexity set to 1 (Low)");
            });
        }

        if (btnHigh) {
            btnHigh.addEventListener('click', () => {
                localStorage.setItem('modelComplexity', '2');
                setComplexityUI(2);
                this.holistic.setOptions({ modelComplexity: 2 });
                console.log("[AI] MediaPipe Holistic Model Complexity set to 2 (High)");
            });
        }

        // Render history on load
        this.renderHistory();

        // Initialize language selection modal logic
        this.initLanguageModal();

        this.camera.start();
        
        this.btnStart.addEventListener('click', () => {
            this.isRecording = !this.isRecording;
            if (this.isRecording) {
                this.btnStart.classList.replace('bg-brand-blue', 'bg-red-500');
                this.btnStart.classList.replace('hover:bg-brand-dark', 'hover:bg-red-600');
                this.btnStart.querySelector('i').classList.add('text-white');
                this.startText.innerText = "Stop Translation";
                this.sequenceBuffer = [];
                if (this.sentenceWords.length > 0) {
                    this.translationText.innerHTML = `${this.sentenceWords.join(' ')} <span class="text-brand-blue animate-pulse">...</span>`;
                } else {
                    this.translationText.innerHTML = `<span class="text-brand-blue animate-pulse">Listening...</span>`;
                }
            } else {
                this.btnStart.classList.replace('bg-red-500', 'bg-brand-blue');
                this.btnStart.classList.replace('hover:bg-red-600', 'hover:bg-brand-dark');
                this.startText.innerText = "Start Translation";
                this.processSignSequence();
            }
        });
    }

    getTopClusters(avgFeat, k = 3) {
        if (!this.clusterModel || !this.clusterModel.centroids) return [];
        const { centroids, scaler_mean, scaler_scale, n_clusters } = this.clusterModel;
        
        // Convert avgFeat (120 dimensions) to 80-dimensional feature vector
        const wristDists = [];
        for (let i = 0; i < 20; i++) {
            const x = avgFeat[3 * i];
            const y = avgFeat[3 * i + 1];
            const z = avgFeat[3 * i + 2];
            wristDists.push(Math.sqrt(x * x + y * y + z * z));
        }
        const remaining = avgFeat.slice(60, 120);
        
        const clusterFeats = new Float32Array(80);
        clusterFeats.set(wristDists, 0);
        clusterFeats.set(remaining, 20);

        if (clusterFeats.length !== scaler_mean.length) return [];

        const scaled = new Float32Array(clusterFeats.length);
        for (let i = 0; i < clusterFeats.length; ++i) {
            scaled[i] = (clusterFeats[i] - scaler_mean[i]) / (scaler_scale[i] + 1e-6);
        }

        const distances = [];
        for (let i = 0; i < n_clusters; ++i) {
            let dist = 0;
            const centroid = centroids[i];
            for (let j = 0; j < scaled.length; ++j) {
                dist += Math.pow(scaled[j] - centroid[j], 2);
            }
            distances.push({ clusterId: i, distance: Math.sqrt(dist) });
        }

        distances.sort((a, b) => a.distance - b.distance);
        return distances.slice(0, k).map(d => d.clusterId);
    }

    onResults(results) {
        this.canvasElement.width = this.videoElement.videoWidth;
        this.canvasElement.height = this.videoElement.videoHeight;
        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        
        if (this.showSkeleton) {
            if (results.leftHandLandmarks) {
                window.drawConnectors(this.canvasCtx, results.leftHandLandmarks, window.HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
                window.drawLandmarks(this.canvasCtx, results.leftHandLandmarks, {color: '#FF0000', lineWidth: 1, radius: 2});
            }
            if (results.rightHandLandmarks) {
                window.drawConnectors(this.canvasCtx, results.rightHandLandmarks, window.HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
                window.drawLandmarks(this.canvasCtx, results.rightHandLandmarks, {color: '#FF0000', lineWidth: 1, radius: 2});
            }

            // Draw Face Spatial Anchors (Yellow)
            if (results.faceLandmarks) {
                const faceIndices = [10, 152, 4, 234, 454, 13, 133, 362];
                this.canvasCtx.fillStyle = '#FFFF00';
                faceIndices.forEach(idx => {
                    const lm = results.faceLandmarks[idx];
                    if (lm) {
                        this.canvasCtx.beginPath();
                        this.canvasCtx.arc(lm.x * this.canvasElement.width, lm.y * this.canvasElement.height, 3, 0, 2 * Math.PI);
                        this.canvasCtx.fill();
                    }
                });
            }

            // Draw Pose Spatial Anchors (Cyan)
            if (results.poseLandmarks) {
                const poseIndices = [7, 8, 11, 12];
                this.canvasCtx.fillStyle = '#00FFFF';
                poseIndices.forEach(idx => {
                    const lm = results.poseLandmarks[idx];
                    if (lm) {
                        this.canvasCtx.beginPath();
                        this.canvasCtx.arc(lm.x * this.canvasElement.width, lm.y * this.canvasElement.height, 3, 0, 2 * Math.PI);
                        this.canvasCtx.fill();
                    }
                });
            }
        }
        
        this.canvasCtx.restore();

        if (this.isRecording) {
            this.captureFrameData(results);
        }
    }

    captureFrameData(results) {
        let nose = null;
        if (results.faceLandmarks && results.faceLandmarks[4]) {
            nose = results.faceLandmarks[4];
        }

        const frame = {
            hands: [],
            face: null,
            pose_anchors: null,
            timestamp: Date.now() / 1000.0
        };

        const packHand = (lms, label) => {
            const wrist = lms[0];
            const frameLms = lms.map(lm => ({
                x: lm.x - wrist.x,
                y: lm.y - wrist.y,
                z: lm.z - wrist.z
            }));

            return {
                label: label,
                landmarks: frameLms,
                wrist_pos: {
                    x: nose ? wrist.x - nose.x : wrist.x,
                    y: nose ? wrist.y - nose.y : wrist.y,
                    z: nose ? wrist.z - nose.z : wrist.z
                }
            };
        };

        if (results.leftHandLandmarks) frame.hands.push(packHand(results.leftHandLandmarks, "Left"));
        if (results.rightHandLandmarks) frame.hands.push(packHand(results.rightHandLandmarks, "Right"));

        // Extract face context (8 Anchors - Relative to Nose)
        if (nose && results.faceLandmarks) {
            const f = results.faceLandmarks;
            const n = nose;
            frame.face = {
                forehead: { x: f[10].x - n.x, y: f[10].y - n.y, z: f[10].z - n.z },
                chin: { x: f[152].x - n.x, y: f[152].y - n.y, z: f[152].z - n.z },
                nose: { x: 0.0, y: 0.0, z: 0.0 },
                l_cheek: { x: f[234].x - n.x, y: f[234].y - n.y, z: f[234].z - n.z },
                r_cheek: { x: f[454].x - n.x, y: f[454].y - n.y, z: f[454].z - n.z },
                mouth: { x: f[13].x - n.x, y: f[13].y - n.y, z: f[13].z - n.z },
                l_eye: { x: f[133].x - n.x, y: f[133].y - n.y, z: f[133].z - n.z },
                r_eye: { x: f[362].x - n.x, y: f[362].y - n.y, z: f[362].z - n.z }
            };

            // Extract pose context (4 Anchors - Relative to Nose)
            if (results.poseLandmarks) {
                const p = results.poseLandmarks;
                frame.pose_anchors = {
                    l_ear: { x: p[7].x - n.x, y: p[7].y - n.y, z: p[7].z - n.z },
                    r_ear: { x: p[8].x - n.x, y: p[8].y - n.y, z: p[8].z - n.z },
                    l_shoulder: { x: p[11].x - n.x, y: p[11].y - n.y, z: p[11].z - n.z },
                    r_shoulder: { x: p[12].x - n.x, y: p[12].y - n.y, z: p[12].z - n.z }
                };
            }
        }

        this.sequenceBuffer.push(frame);
    }

    async processSignSequence() {
        if (this.sequenceBuffer.length === 0) return;
        if (this.sentenceWords.length > 0) {
            this.translationText.innerHTML = `${this.sentenceWords.join(' ')} <span class="text-slate-400 text-lg">...</span>`;
        } else {
            this.translationText.innerHTML = `<span class="text-slate-400 text-lg">Analyzing...</span>`;
        }
        
        // 1. Extract Features for all frames (120 dimensions for DTW/Dynamic RF)
        const features = this.sequenceBuffer.map(f => {
            const feat = extract_ml_features(f);
            return feat ? feat.slice(0, 120) : null;
        }).filter(f => f !== null);

        if (features.length === 0) {
            this.translationText.innerText = "No hands detected.";
            return;
        }

        // 2. Check Movement Variance (Static vs Dynamic)
        let maxHands = 0;
        let maxWristDist = 0.0;
        let maxShapeVariance = 0.0;
        
        let startWrist = {}; // Map label -> wrist_pos
        let startShape = {}; // Map label -> landmarks
        let initDone = false;

        for (const f of this.sequenceBuffer) {
            let frameHands = f.hands.length;
            maxHands = Math.max(maxHands, frameHands);
            
            if (!initDone && frameHands > 0) {
                for (const h of f.hands) {
                    startWrist[h.label] = { x: h.wrist_pos.x, y: h.wrist_pos.y, z: h.wrist_pos.z };
                    startShape[h.label] = h.landmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }));
                }
                initDone = true;
            }

            if (initDone) {
                for (const h of f.hands) {
                    const startW = startWrist[h.label];
                    const startS = startShape[h.label];
                    if (startW && startS) {
                        let dx = h.wrist_pos.x - startW.x;
                        let dy = h.wrist_pos.y - startW.y;
                        let dz = h.wrist_pos.z - startW.z;
                        maxWristDist = Math.max(maxWristDist, Math.sqrt(dx*dx + dy*dy + dz*dz));

                        for (let j = 0; j < h.landmarks.length && j < startS.length; ++j) {
                            let sx = h.landmarks[j].x - startS[j].x;
                            let sy = h.landmarks[j].y - startS[j].y;
                            let sz = h.landmarks[j].z - startS[j].z;
                            maxShapeVariance = Math.max(maxShapeVariance, Math.sqrt(sx*sx + sy*sy + sz*sz));
                        }
                    }
                }
            }
        }

        const TH_WRIST = 0.20;
        const TH_SHAPE = 0.12;
        const isDynamic = (this.sequenceBuffer.length >= 12) && (maxWristDist > TH_WRIST || maxShapeVariance > TH_SHAPE);

        let bestLabel = "";
        let routeDecision = "";
        let debugHtml = "";
        let topCandidates = []; // Top-3 raw sign labels from this recognition

        if (isDynamic) {
            // DYNAMIC SIGN (Tri-Factor Fusion Engine)
            routeDecision = "DYNAMIC (Tri-Factor Fusion)";
            console.log(`Routing to Dynamic Engine... (Max Wrist Dist: ${maxWristDist.toFixed(2)}, Max Shape Var: ${maxShapeVariance.toFixed(2)}, Max Hands: ${maxHands})`);
            
            // 1. Target hand-count category
            let twoHandFrames = 0;
            for (const f of this.sequenceBuffer) {
                if (f.hands.length >= 2) twoHandFrames++;
            }
            const targetCat = twoHandFrames >= 5 ? "movement/2_hands" : "movement/single_hand";
            console.log(`Target category: ${targetCat} (2-hand frames: ${twoHandFrames})`);

            // 2. ML Shape Shortlist (using dynamicModel)
            const shapeVotes = {};
            if (this.dynamicModel) {
                for (let i = 0; i < this.sequenceBuffer.length; ++i) {
                    if (i % 4 === 0 || i === 10 || i === 15) {
                        const frame = this.sequenceBuffer[i];
                        const fullFeat = extract_ml_features(frame);
                        if (fullFeat && fullFeat.length >= 120) {
                            const dynFeat = fullFeat.slice(0, 120);
                            const probs = this.dynamicModel.predict_proba(dynFeat);
                            const classes = this.dynamicModel.classes;
                            const weight = (i === 10 || i === 15) ? 5.0 : 1.0;
                            for (let k = 0; k < probs.length; ++k) {
                                const cls = classes[k];
                                shapeVotes[cls] = (shapeVotes[cls] || 0) + probs[k] * weight;
                            }
                        }
                    }
                }
            }

            // Normalise votes
            let sortedVotes = [];
            let totalVotes = 0;
            for (const cls in shapeVotes) {
                totalVotes += shapeVotes[cls];
            }
            if (totalVotes < 0.1) totalVotes = 1.0;

            for (const cls in shapeVotes) {
                sortedVotes.push({ class: cls, score: shapeVotes[cls], confidence: shapeVotes[cls] / totalVotes });
            }
            sortedVotes.sort((a, b) => b.score - a.score);

            // 3. Cluster Pruning (using K-Means clusterModel)
            let allowedClusters = [];
            if (this.clusterModel) {
                const avgFeat = new Float32Array(120);
                let count = 0;
                const startF = (features.length > 10) ? 5 : 0;
                for (let i = startF; i < features.length; ++i) {
                    for (let k = 0; k < 120; ++k) {
                        avgFeat[k] += features[i][k];
                    }
                    count++;
                }
                if (count > 0) {
                    for (let k = 0; k < 120; ++k) {
                        avgFeat[k] /= count;
                    }
                }
                allowedClusters = this.getTopClusters(avgFeat, 3);
                console.log(`Allowed clusters (top 3):`, allowedClusters);
            }

            // 4. Fusion Candidates Scoring
            const candidates = [];
            const topN = Math.min(5, sortedVotes.length);
            const currentMlPower = 0.75; // Matches current_ml_power in C++

            for (let i = 0; i < topN; ++i) {
                const name = sortedVotes[i].class;
                const confidence = sortedVotes[i].confidence;
                if (confidence < 0.001) continue;

                let clusterBonus = 0.0;
                if (this.clusterModel && allowedClusters.length > 0 && this.fileClusterMapping[name] !== undefined) {
                    const signCluster = this.fileClusterMapping[name];
                    if (signCluster === allowedClusters[0]) clusterBonus = 0.30;
                    else if (allowedClusters.length > 1 && signCluster === allowedClusters[1]) clusterBonus = 0.24;
                    else if (allowedClusters.length > 2 && signCluster === allowedClusters[2]) clusterBonus = 0.19;
                }

                const mlBonus = Math.sqrt(confidence) * 1.50;
                const totalBonus = Math.min(0.95, mlBonus + clusterBonus);

                // Compute DTW distance for this candidate against all matched templates
                let dtwDist = 999.0;
                let found = false;
                let actualFolder = "";

                // Look for template in our loaded templates database
                for (const tpl of this.templates) {
                    if (tpl.label === name) {
                        dtwDist = DtwEngine.computeDualScore(features, tpl.features, 0.4);
                        found = true;
                        actualFolder = tpl.category;
                        break;
                    }
                }

                if (found) {
                    const movementScore = dtwDist;
                    const aiScore = (1.0 - confidence) * 50.0 - (clusterBonus * 30.0);
                    const fused = ((1.0 - currentMlPower) * movementScore) + (currentMlPower * aiScore);
                    candidates.push({
                        name: name,
                        fusedScore: fused,
                        dtwDist: dtwDist,
                        category: actualFolder,
                        confidence: confidence,
                        clusterBonus: clusterBonus,
                        totalBonus: totalBonus
                    });
                }
            }

            // Sort by fused score ascending (lower is better)
            candidates.sort((a, b) => a.fusedScore - b.fusedScore);

            // Filter to matching hand count category
            const prunedList = candidates.filter(c => c.category === targetCat);
            if (prunedList.length > 0) {
                bestLabel = prunedList[0].name;
            } else if (candidates.length > 0) {
                bestLabel = candidates[0].name;
                console.warn(`No templates found matching target category ${targetCat} in ML shortlist. Falling back to raw winner.`);
            } else if (sortedVotes.length > 0) {
                bestLabel = sortedVotes[0].class;
                console.warn(`No templates found for any shortlist candidates. Falling back to top ML class: ${bestLabel}`);
            } else {
                bestLabel = "None";
            }

            // Capture top-3 candidates for history
            const activeList = prunedList.length > 0 ? prunedList : candidates;
            topCandidates = activeList.slice(0, 3).map(c => ({
                name: c.name,
                score: c.fusedScore,
                dtwDist: c.dtwDist,
                category: c.category,
                isDynamic: true
            }));
            if (topCandidates.length === 0) {
                topCandidates = sortedVotes.slice(0, 3).map(v => ({
                    name: v.class,
                    score: v.confidence,
                    dtwDist: null,
                    category: null,
                    isDynamic: true
                }));
            }

            // Build detailed developer diagnostics output
            debugHtml = `
              <div class="space-y-1">
                <p><span class="text-slate-400">Route:</span> <span class="text-green-400 font-bold">${routeDecision}</span></p>
                <p><span class="text-slate-400">Total Frames:</span> ${this.sequenceBuffer.length}</p>
                <p><span class="text-slate-400">Max Wrist Dist:</span> ${maxWristDist.toFixed(3)} (TH: ${TH_WRIST})</p>
                <p><span class="text-slate-400">Max Shape Var:</span> ${maxShapeVariance.toFixed(3)} (TH: ${TH_SHAPE})</p>
                <p><span class="text-slate-400">Target Category:</span> <span class="text-indigo-300 font-semibold">${targetCat}</span></p>
                <p><span class="text-slate-400">Candidate Clusters:</span> ${allowedClusters.join(', ')}</p>
                <div class="border-t border-slate-700 mt-2 pt-2">
                  <p class="font-bold text-slate-300">Top 5 ML Shortlist:</p>
                  ${sortedVotes.slice(0, 5).map(v => `
                    <p><span class="text-yellow-300">${v.class}</span>: ${(v.confidence * 100).toFixed(1)}%</p>
                  `).join('')}
                </div>
                <div class="border-t border-slate-700 mt-2 pt-2">
                  <p class="font-bold text-slate-300">Top Fused Candidates (Pruned):</p>
                  ${prunedList.slice(0, 3).map((c, idx) => `
                    <p>${idx + 1}. <span class="text-yellow-300">${c.name}</span> - Fused: <span class="text-cyan-300">${c.fusedScore.toFixed(2)}</span> (DTW: ${c.dtwDist.toFixed(2)})</p>
                  `).join('')}
                </div>
                <div class="border-t border-slate-700 mt-2 pt-2">
                  <p class="font-bold text-slate-300">Top Fused Candidates (Raw):</p>
                  ${candidates.slice(0, 3).map((c, idx) => `
                    <p>${idx + 1}. <span class="text-yellow-300">${c.name}</span> - Fused: <span class="text-cyan-300">${c.fusedScore.toFixed(2)}</span> (DTW: ${c.dtwDist.toFixed(2)})</p>
                  `).join('')}
                </div>
              </div>
            `;
        } else {
            // STATIC SIGN (Random Forest)
            routeDecision = "STATIC (Random Forest)";
            console.log(`Routing to Static Model... (Max Wrist Dist: ${maxWristDist.toFixed(2)}, Max Shape Var: ${maxShapeVariance.toFixed(2)}, Max Hands: ${maxHands})`);
            
            let confidence = 0.0;
            if (this.staticModel) {
                const midIndex = Math.floor(this.sequenceBuffer.length / 2);
                const midFrame = this.sequenceBuffer[midIndex];
                const fullFeatures = extract_ml_features(midFrame);
                if (fullFeatures && fullFeatures.length === 142) {
                    const result = this.staticModel.predict(fullFeatures);
                    bestLabel = result.label;
                    confidence = result.confidence;

                    // --- THE SPATIAL GUARD ---
                    const idx_r = fullFeatures[120];
                    const mid_r = fullFeatures[121];
                    const avg_angle = (idx_r + mid_r) / 2.0;

                    if (avg_angle > 0.65) { // Sideways Hand
                        if (bestLabel === "R" || bestLabel === "U" || bestLabel === "V" || bestLabel === "I") {
                            bestLabel = "H"; // Force H
                        }
                    } else if (avg_angle < 0.35) { // Vertical Hand
                        if (bestLabel === "H") {
                            bestLabel = "U"; // Force U
                        }
                    }

                    // Capture top-3 candidates for history from RF proba scores
                    const allProbs = this.staticModel.predict_proba(fullFeatures);
                    const classProbs = Array.from(allProbs).map((p, i) => ({ cls: this.staticModel.classes[i], prob: p }));
                    classProbs.sort((a, b) => b.prob - a.prob);
                    topCandidates = classProbs.slice(0, 3).map(cp => ({
                        name: cp.cls,
                        score: cp.prob,
                        dtwDist: null,
                        category: null,
                        isDynamic: false
                    }));
                }
            }

            debugHtml = `
              <div class="space-y-1">
                <p><span class="text-slate-400">Route:</span> <span class="text-blue-400 font-bold">${routeDecision}</span></p>
                <p><span class="text-slate-400">Total Frames:</span> ${this.sequenceBuffer.length}</p>
                <p><span class="text-slate-400">Max Wrist Dist:</span> ${maxWristDist.toFixed(3)} (TH: ${TH_WRIST})</p>
                <p><span class="text-slate-400">Max Shape Var:</span> ${maxShapeVariance.toFixed(3)} (TH: ${TH_SHAPE})</p>
                <p><span class="text-slate-400">Max Hands:</span> ${maxHands}</p>
                <div class="border-t border-slate-700 mt-2 pt-2">
                  <p class="font-bold text-slate-300">RF Prediction:</p>
                  <p>Winner: <span class="text-yellow-300">${bestLabel}</span></p>
                  <p>Confidence: <span class="text-cyan-300">${(confidence * 100).toFixed(1)}%</span></p>
                </div>
              </div>
            `;
        }

        // Update debug panel content if it exists
        const debugContent = document.getElementById('debug-content');
        if (debugContent) {
            debugContent.innerHTML = debugHtml;
        }

        // Clean label suffixes like _Tonkla, 1, 2, 3, etc.
        const cleanLabel = bestLabel.replace(/_?[tT]onkla\d*/g, '').replace(/_?\d+/g, '');

        // Track top-3 candidates (cleaned) for history — always put chosen first
        if (cleanLabel && cleanLabel !== 'None' && cleanLabel !== '') {
            const seen = new Set();
            const cleanChosen = cleanLabel;
            
            // Map and clean topCandidates
            const cleanedCandidates = topCandidates.map(tc => {
                const cleanedName = tc.name.replace(/_?[tT]onkla\d*/g, '').replace(/_?\d+/g, '');
                return {
                    ...tc,
                    name: cleanedName
                };
            }).filter(tc => tc.name && tc.name !== 'None');

            // Find if chosen exists in cleanedCandidates, or create a placeholder
            let chosenObj = cleanedCandidates.find(c => c.name === cleanChosen);
            if (!chosenObj) {
                chosenObj = {
                    name: cleanChosen,
                    score: 0,
                    dtwDist: null,
                    category: null,
                    isDynamic: isDynamic
                };
            }

            const formattedCandidates = [chosenObj];
            seen.add(cleanChosen);

            for (const tc of cleanedCandidates) {
                if (!seen.has(tc.name)) {
                    seen.add(tc.name);
                    formattedCandidates.push(tc);
                }
            }

            this.wordCandidates.push({
                candidates: formattedCandidates.slice(0, 3),
                chosen: cleanChosen,
                isDynamic: isDynamic
            });
        }

        // 3. Display word (keep per-word capitalization, skip heavy polishing)
        if (this.sentenceWords.length > 0) {
            this.translationText.innerHTML = `${this.sentenceWords.join(' ')} <span class="text-brand-blue animate-pulse">...</span>`;
        } else {
            this.translationText.innerHTML = `<span class="text-brand-blue animate-pulse">Polishing Grammar...</span>`;
        }
        
        try {
            const finalWord = await this.polishWithAI(cleanLabel);
            this.sentenceWords.push(finalWord);
            this.translationText.innerText = this.sentenceWords.join(' ');
        } catch(e) {
            this.sentenceWords.push(cleanLabel);
            this.translationText.innerText = this.sentenceWords.join(' ');
        }

        this.updateTranslateButton();
    }

    async polishWithAI(rawLabel) {
        const dictionary = {
            "hello": "Hello",
            "thank_you": "Thank you",
            "i_love_you": "I love you",
            "eat": "Eat"
        };
        return new Promise(resolve => {
            setTimeout(() => {
                const polished = dictionary[rawLabel.toLowerCase()] ||
                    rawLabel.charAt(0).toUpperCase() + rawLabel.slice(1).replace(/_/g, ' ');
                resolve(polished);
            }, 200);
        });
    }

    updateTranslateButton() {
        const wrapper = document.getElementById('translate-btn-wrapper');
        if (wrapper) wrapper.classList.toggle('hidden', this.sentenceWords.length === 0);
    }

    async getActiveGeminiModel(apiKey) {
        let cached = localStorage.getItem('geminiActiveModel');
        if (cached) {
            return cached;
        }

        console.log("[AI] Discovering and checking working models...");
        const listUrl = `https://generativelanguage.googleapis.com/v1/models?key=${apiKey}`;
        
        try {
            const listRes = await fetch(listUrl);
            if (!listRes.ok) {
                throw new Error(`Failed to list models: HTTP ${listRes.status}`);
            }
            const listData = await listRes.json();
            const models = listData.models || [];
            
            // Filter models that support generateContent
            const candidateModels = models
                .filter(m => m.supportedGenerationMethods && m.supportedGenerationMethods.includes('generateContent'))
                .map(m => m.name);

            if (candidateModels.length === 0) {
                throw new Error("No models supporting generateContent found.");
            }

            console.log(`[AI] Testing ${candidateModels.length} models...`);
            
            // Function to check a single model
            const checkModel = async (modelName) => {
                const testUrl = `https://generativelanguage.googleapis.com/v1/${modelName}:generateContent?key=${apiKey}`;
                const payload = { contents: [{ parts: [{ text: "hi" }] }] };
                
                // Add a timeout of 5 seconds
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 5000);
                
                try {
                    const r = await fetch(testUrl, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                        signal: controller.signal
                    });
                    clearTimeout(timeoutId);
                    
                    if (r.ok) {
                        return { modelName, working: true, status: "OK" };
                    } else {
                        const errBody = await r.json().catch(() => ({}));
                        const errMsg = errBody?.error?.message || `HTTP ${r.status}`;
                        return { modelName, working: false, status: `Failed (${r.status}: ${errMsg})` };
                    }
                } catch (e) {
                    clearTimeout(timeoutId);
                    return { modelName, working: false, status: `Failed (${e.name}: ${e.message})` };
                }
            };

            // Run checks concurrently
            const results = await Promise.all(candidateModels.map(name => checkModel(name)));
            const workingModels = [];
            
            // Print status of each checked model
            let debugHtml = `<div class="space-y-1 mt-2 pt-2 border-t border-slate-700">
                <p class="font-bold text-slate-300">Model Verification Checks:</p>`;
            
            for (const res of results) {
                console.log(`  - ${res.modelName}: ${res.status}`);
                debugHtml += `<p class="text-[10px]"><span class="text-slate-400">${res.modelName.replace('models/', '')}:</span> <span class="${res.working ? 'text-green-400' : 'text-red-400'}">${res.status}</span></p>`;
                if (res.working) {
                    workingModels.push(res.modelName);
                }
            }
            debugHtml += `</div>`;
            
            // Also append to the Developer Diagnostics panel
            const debugPanelContent = document.getElementById('debug-content');
            if (debugPanelContent) {
                debugPanelContent.innerHTML += debugHtml;
            }

            if (workingModels.length === 0) {
                throw new Error("Checked all models but none of them are working/authorized.");
            }

            const getModelRank = (modelName) => {
                const modelLower = modelName.toLowerCase();
                if (modelLower.includes("gemini-3.5-flash")) return 0;
                if (modelLower.includes("gemini-3.1-flash")) return 1;
                if (modelLower.includes("gemini-2.5-flash")) return 2;
                if (modelLower.includes("gemini-2.0-flash")) return 3;
                if (modelLower.includes("gemini-1.5-flash")) return 4;
                if (modelLower.includes("flash")) return 5;
                if (modelLower.includes("gemini")) return 6;
                return 7;
            };

            workingModels.sort((a, b) => getModelRank(a) - getModelRank(b));
            const selectedModel = workingModels[0];
            console.log(`[AI] Selected model: ${selectedModel}`);
            localStorage.setItem('geminiActiveModel', selectedModel);
            return selectedModel;

        } catch (err) {
            console.error("[AI ERROR] Model check failed:", err);
            throw err;
        }
    }

    async translateSentence() {
        const apiKey = localStorage.getItem('geminiApiKey') || '';
        if (!apiKey) {
            this.translationText.innerHTML = `<span class="text-red-400 text-sm font-medium">⚠ Please add your Gemini API key in Settings first.</span>`;
            return;
        }
        const words = [...this.sentenceWords];
        if (words.length === 0) return;

        const btn = document.getElementById('btn-ai-translate');
        if (btn) {
            btn.innerHTML = `<i class="ph ph-spinner-gap text-lg animate-spin"></i><span>Translating...</span>`;
            btn.disabled = true;
        }
        this.translationText.innerHTML = `<span class="text-slate-400 text-sm animate-pulse">Building sentence with Gemini...</span>`;

        try {
            // Check all models to select the best active one (uses cache if available)
            const targetModel = await this.getActiveGeminiModel(apiKey);
            const targetLang = this.selectedTargetLang || 'English';
            
            let prompt = "";
            if (targetLang === 'English') {
                prompt = `You are a sign language interpreter. These words were recognized from sign language gestures in order: "${words.join(', ')}". Construct the most natural, grammatically correct English sentence from these words. Output only the final sentence, nothing else.`;
            } else {
                prompt = `You are a sign language interpreter. These words were recognized from sign language gestures in order: "${words.join(', ')}". First, construct the most natural grammatically correct English sentence from these words. Second, translate that polished English sentence into the target spoken language "${targetLang}". Output only the final translated "${targetLang}" sentence, nothing else.`;
            }

            const res = await fetch(
                `https://generativelanguage.googleapis.com/v1/${targetModel}:generateContent?key=${apiKey}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] })
                }
            );
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err?.error?.message || `HTTP ${res.status}`);
            }
            const data = await res.json();
            const sentence = data.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
            if (!sentence) throw new Error('Empty response from Gemini');

            this.translationText.innerHTML = `
                <span class="text-violet-500 text-[10px] font-bold uppercase tracking-widest block mb-1">✦ AI Sentence</span>
                ${sentence}
            `;

            this.saveToHistory(words, sentence, JSON.parse(JSON.stringify(this.wordCandidates)));
            this.wordCandidates = [];

        } catch (err) {
            console.error('Gemini error:', err);
            this.translationText.innerHTML = `<span class="text-red-400 text-sm">Error: ${err.message}</span>`;
            // Clear cached model on failure in case it became stale or failed
            localStorage.removeItem('geminiActiveModel');
        } finally {
            if (btn) {
                btn.innerHTML = `<i class="ph ph-sparkle text-lg"></i><span>Translate to Sentence</span><i class="ph ph-arrow-right text-sm"></i>`;
                btn.disabled = false;
            }
        }
    }

    saveToHistory(rawWords, aiSentence, wordCandidates) {
        const item = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            rawWords: rawWords.join(' '),
            aiSentence,
            wordCandidates
        };
        this.history.unshift(item);
        this.history = this.history.slice(0, 50);
        localStorage.setItem('signHistory', JSON.stringify(this.history));
        this.renderHistory();
    }

    renderHistory() {
        const list = document.getElementById('history-list');
        if (!list) return;

        if (this.history.length === 0) {
            list.innerHTML = `
                <div class="flex flex-col items-center justify-center py-16 text-center">
                    <i class="ph ph-clock-counter-clockwise text-5xl text-slate-200 mb-3"></i>
                    <p class="text-slate-400 font-medium">No history yet</p>
                    <p class="text-slate-300 text-sm mt-1">Translate a sign session to see it here</p>
                </div>`;
            return;
        }

        list.innerHTML = this.history.map(item => {
            const d = new Date(item.timestamp);
            const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const dateStr = d.toLocaleDateString([], { month: 'short', day: 'numeric' });
            const wc = item.wordCandidates.length;
            return `
                <div class="history-card bg-white border border-slate-200 rounded-2xl p-4 shadow-sm cursor-pointer hover:border-brand-blue hover:shadow-md transition-all duration-200 active:scale-[0.98]" data-id="${item.id}">
                    <div class="flex items-start gap-3">
                        <div class="w-9 h-9 bg-gradient-to-br from-violet-100 to-purple-100 rounded-xl flex items-center justify-center shrink-0">
                            <i class="ph ph-sparkle text-violet-500 text-base"></i>
                        </div>
                        <div class="flex-1 min-w-0">
                            <p class="text-sm font-semibold text-slate-800 leading-snug">${item.aiSentence}</p>
                            <p class="text-xs text-slate-400 mt-1 truncate">${item.rawWords}</p>
                            <div class="flex items-center gap-2 mt-1.5">
                                <span class="text-[10px] text-slate-300">${dateStr} · ${timeStr}</span>
                                <span class="text-[10px] bg-slate-100 text-slate-400 px-1.5 py-0.5 rounded-full">${wc} sign${wc !== 1 ? 's' : ''}</span>
                            </div>
                        </div>
                        <i class="ph ph-caret-right text-slate-300 text-lg mt-0.5 shrink-0"></i>
                    </div>
                </div>`;
        }).join('');

        list.querySelectorAll('.history-card').forEach(card => {
            card.addEventListener('click', () => {
                const item = this.history.find(h => h.id === parseInt(card.dataset.id));
                if (item) this.openHistoryDetail(item);
            });
        });
    }

    openHistoryDetail(item) {
        const panel = document.getElementById('history-detail');
        const content = document.getElementById('history-detail-content');
        if (!panel || !content) return;

        const dateObj = new Date(item.timestamp);
        const formattedDate = dateObj.toLocaleDateString([], { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
        const formattedTime = dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        content.innerHTML = `
            <!-- AI Reconstructed Sentence with solid opaque premium background -->
            <div class="bg-gradient-to-r from-violet-600 to-indigo-600 dark:from-violet-700 dark:to-indigo-800 text-white rounded-2xl p-5 shadow-lg border-0">
                <div class="flex items-center gap-2 mb-2">
                    <i class="ph ph-sparkle text-white text-sm animate-pulse"></i>
                    <span class="text-[10px] font-bold text-white/90 uppercase tracking-widest">AI Polished Sentence</span>
                </div>
                <p class="text-xl font-bold leading-snug">${item.aiSentence}</p>
            </div>

            <!-- Original Raw Sentence block -->
            <div class="bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-sm">
                <div class="flex items-center gap-2 mb-1.5">
                    <i class="ph ph-text-t text-slate-500 dark:text-slate-400 text-sm"></i>
                    <span class="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Before Reconstruction (Raw Signs)</span>
                </div>
                <p class="text-base font-semibold text-slate-800 dark:text-slate-200 italic">"${item.rawWords}"</p>
            </div>

            <!-- Prediction Analysis Title -->
            <div class="flex items-center justify-between mt-6 mb-2">
                <div class="flex items-center gap-2">
                    <i class="ph ph-presentation-chart text-brand-blue dark:text-sky-400 text-xl"></i>
                    <h4 class="text-sm font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider">Prediction Analysis</h4>
                </div>
                <i class="ph ph-info text-slate-400 dark:text-slate-500 text-lg cursor-pointer" title="Displays scoring criteria and classification tags for each sign."></i>
            </div>

            <!-- Detailed word candidates list -->
            <div class="space-y-4">
                ${item.wordCandidates.map((wc, i) => {
                    const isDynamic = wc.isDynamic;
                    const typeLabel = isDynamic ? "Dynamic (DTW + RF)" : "Static (Random Forest)";
                    
                    return `
                    <div class="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-2xl p-4 shadow-sm space-y-3">
                        <div class="flex items-center justify-between border-b border-slate-100 dark:border-slate-700 pb-2">
                            <span class="text-xs font-bold text-slate-400 dark:text-slate-500">Word ${i + 1}</span>
                            <span class="text-[10px] bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400 px-2 py-0.5 rounded-full font-semibold">${typeLabel}</span>
                        </div>
                        
                        <p class="text-xl font-bold text-brand-blue dark:text-sky-400">'${wc.chosen}'</p>
                        
                        <div class="space-y-2">
                            <p class="text-[11px] font-bold text-slate-500 dark:text-slate-400">Top Predictions:</p>
                            <div class="grid grid-cols-1 gap-2">
                                ${wc.candidates.map((c, ci) => {
                                    const isActive = (c.name === wc.chosen);
                                    let handTag = "";
                                    if (c.category) {
                                        handTag = c.category.includes("2_hands") ? " [2H]" : " [1H]";
                                    }
                                    
                                    let scoreStr = "";
                                    if (isDynamic) {
                                        scoreStr = `Fused: ${c.score !== null && c.score !== undefined ? c.score.toFixed(2) : '0.00'}`;
                                        if (c.dtwDist !== null && c.dtwDist !== undefined) {
                                            scoreStr += ` (DTW: ${c.dtwDist.toFixed(2)})`;
                                        }
                                    } else {
                                        scoreStr = `Confidence: ${(c.score * 100).toFixed(1)}%`;
                                    }
                                    
                                    return `
                                    <div class="flex items-center justify-between p-2.5 rounded-xl border ${
                                        isActive 
                                            ? 'bg-brand-blue/10 border-brand-blue dark:bg-sky-500/10 dark:border-sky-500' 
                                            : 'bg-slate-50 dark:bg-slate-900 border-slate-100 dark:border-slate-800'
                                    }">
                                        <div class="flex items-center gap-2">
                                            <span class="text-xs font-bold ${isActive ? 'text-brand-blue dark:text-sky-400' : 'text-slate-400'}">${ci + 1}.</span>
                                            <span class="text-sm font-bold ${isActive ? 'text-brand-blue dark:text-sky-400' : 'text-slate-700 dark:text-slate-300'}">
                                                ${c.name}${handTag}
                                            </span>
                                        </div>
                                        <span class="text-xs font-mono text-slate-500 dark:text-slate-400">${scoreStr}</span>
                                    </div>
                                    `;
                                }).join('')}
                            </div>
                        </div>
                    </div>
                    `;
                }).join('')}
            </div>

            <!-- Footer date and time -->
            <div class="text-center pt-4 border-t border-slate-100 dark:border-slate-700">
                <p class="text-[10px] text-slate-400 dark:text-slate-500 font-semibold">${formattedDate}</p>
                <p class="text-[9px] text-slate-300 dark:text-slate-600 mt-0.5">${formattedTime}</p>
            </div>
        `;

        panel.classList.remove('translate-y-full');
    }

    initLanguageModal() {
        const btnSource = document.getElementById('btn-lang-source');
        const btnTarget = document.getElementById('btn-lang-target');
        const modalBg = document.getElementById('modal-language-bg');
        const btnClose = document.getElementById('btn-close-lang');
        const tabSign = document.getElementById('tab-sign-lang');
        const tabSpoken = document.getElementById('tab-spoken-lang');
        const btnConfirm = document.getElementById('btn-confirm-lang');
        const searchInput = document.getElementById('lang-search');

        if (btnSource) {
            btnSource.addEventListener('click', () => {
                this.openLanguageModal('source');
            });
        }

        if (btnTarget) {
            btnTarget.addEventListener('click', () => {
                this.openLanguageModal('target');
            });
        }

        if (modalBg) {
            modalBg.addEventListener('click', () => this.closeLanguageModal());
        }

        if (btnClose) {
            btnClose.addEventListener('click', () => this.closeLanguageModal());
        }

        if (tabSign) {
            tabSign.addEventListener('click', () => this.switchModalTab('sign'));
        }

        if (tabSpoken) {
            tabSpoken.addEventListener('click', () => this.switchModalTab('spoken'));
        }

        if (btnConfirm) {
            btnConfirm.addEventListener('click', () => this.confirmLanguageSelection());
        }

        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.renderModalLangList(e.target.value);
            });
        }
    }

    openLanguageModal(context) {
        this.modalContext = context;
        this.currentModalTab = context === 'source' ? 'sign' : 'spoken';
        this.tempSelectedLang = context === 'source' ? this.selectedSourceLang : this.selectedTargetLang;

        const modalWrap = document.getElementById('modal-language');
        const modalBg = document.getElementById('modal-language-bg');
        const modalContent = document.getElementById('modal-language-content');
        const searchInput = document.getElementById('lang-search');

        if (searchInput) searchInput.value = '';

        this.switchModalTab(this.currentModalTab);

        if (modalWrap && modalBg && modalContent) {
            modalWrap.classList.remove('pointer-events-none');
            modalBg.classList.replace('opacity-0', 'opacity-100');
            modalContent.classList.replace('translate-y-full', 'translate-y-0');
        }
    }

    closeLanguageModal() {
        const modalWrap = document.getElementById('modal-language');
        const modalBg = document.getElementById('modal-language-bg');
        const modalContent = document.getElementById('modal-language-content');

        if (modalWrap && modalBg && modalContent) {
            modalBg.classList.replace('opacity-100', 'opacity-0');
            modalContent.classList.replace('translate-y-0', 'translate-y-full');
            setTimeout(() => modalWrap.classList.add('pointer-events-none'), 300);
        }
    }

    switchModalTab(tabType) {
        const tabSign = document.getElementById('tab-sign-lang');
        const tabSpoken = document.getElementById('tab-spoken-lang');
        const listContainer = document.getElementById('modal-lang-list');
        const searchInput = document.getElementById('lang-search');

        const query = searchInput ? searchInput.value : '';

        if (tabType === 'sign') {
            if (tabSign) {
                tabSign.className = "flex-1 py-2 text-sm font-semibold text-brand-blue border-b-2 border-brand-blue transition-all duration-300";
            }
            if (tabSpoken) {
                tabSpoken.className = "flex-1 py-2 text-sm font-semibold text-slate-400 transition-all duration-300";
            }
        } else {
            if (tabSign) {
                tabSign.className = "flex-1 py-2 text-sm font-semibold text-slate-400 transition-all duration-300";
            }
            if (tabSpoken) {
                tabSpoken.className = "flex-1 py-2 text-sm font-semibold text-brand-blue border-b-2 border-brand-blue transition-all duration-300";
            }
        }

        this.currentModalTab = tabType;

        // Beautiful fade-out/fade-in animation for list content
        if (listContainer) {
            listContainer.classList.add('opacity-0');
            setTimeout(() => {
                this.renderModalLangList(query);
                listContainer.classList.remove('opacity-0');
            }, 150);
        }
    }

    renderModalLangList(searchQuery = '') {
        const listContainer = document.getElementById('modal-lang-list');
        if (!listContainer) return;

        const langs = this.currentModalTab === 'sign' ? this.signLanguages : this.spokenLanguages;
        
        listContainer.innerHTML = '';
        const query = searchQuery.toLowerCase().trim();

        langs.forEach(lang => {
            if (query && !lang.toLowerCase().includes(query)) return;

            const isSelected = lang === this.tempSelectedLang;
            const optionDiv = document.createElement('div');
            optionDiv.className = `lang-option flex items-center justify-between p-3 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 cursor-pointer transition-all duration-150`;

            optionDiv.innerHTML = `
                <span class="text-sm font-medium ${isSelected ? 'text-brand-blue font-semibold dark:text-sky-400' : 'text-slate-800 dark:text-slate-200'}">${lang}</span>
                ${isSelected 
                    ? '<i class="ph-fill ph-check-circle text-brand-blue dark:text-sky-400 text-xl"></i>' 
                    : '<div class="w-5 h-5 rounded-full border-2 border-slate-300 dark:border-slate-600 transition-all"></div>'
                }
            `;

            optionDiv.addEventListener('click', () => {
                this.tempSelectedLang = lang;
                this.renderModalLangList(searchQuery); // re-render to update selection style
            });

            listContainer.appendChild(optionDiv);
        });

        if (listContainer.children.length === 0) {
            listContainer.innerHTML = `
                <div class="text-center py-8 text-slate-400 dark:text-slate-500 text-sm">
                    No languages found
                </div>
            `;
        }
    }

    confirmLanguageSelection() {
        if (!this.tempSelectedLang) {
            this.closeLanguageModal();
            return;
        }

        if (this.modalContext === 'source') {
            this.selectedSourceLang = this.tempSelectedLang;
            localStorage.setItem('selectedSourceLang', this.selectedSourceLang);
            const btnSource = document.getElementById('btn-lang-source');
            if (btnSource) btnSource.innerText = this.selectedSourceLang;
        } else {
            this.selectedTargetLang = this.tempSelectedLang;
            localStorage.setItem('selectedTargetLang', this.selectedTargetLang);
            const btnTarget = document.getElementById('btn-lang-target');
            if (btnTarget) btnTarget.innerText = this.selectedTargetLang;
        }

        this.closeLanguageModal();
    }
}
