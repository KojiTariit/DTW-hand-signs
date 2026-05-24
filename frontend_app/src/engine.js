// engine.js - Pure JavaScript implementation of the C++ DTW & Random Forest Hybrid Engine

// Helper Math Functions
function magnitude(v) {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

function sub_points(a, b) {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function dot_product(v1, v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

function dist3D(a, b) {
    return Math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2);
}

// ---------------------------------------------------------
// 1. ML Feature Extraction (142 Features)
// ---------------------------------------------------------
export function extract_ml_features(frame) {
    const features = [];
    const hand = frame.hands.find(h => h.landmarks && h.landmarks.length === 21);
    if (!hand) return null;

    const lms = hand.landmarks;
    const p0 = lms[0];
    const p9 = lms[9];
    let hand_size = magnitude(sub_points(p9, p0));
    if (hand_size < 1e-6) hand_size = 1.0;

    // 1. Wrist-Relative XYZ (60)
    for (let i = 1; i < 21; ++i) {
        let d = sub_points(lms[i], p0);
        features.push(d.x / hand_size, d.y / hand_size, d.z / hand_size);
    }

    // 2. Finger Extension Ratios (5)
    const curltips = [4, 8, 12, 16, 20];
    const curlmcps = [2, 5, 9, 13, 17];
    for (let i = 0; i < 5; ++i) {
        let dist_tip_mcp = magnitude(sub_points(lms[curltips[i]], lms[curlmcps[i]]));
        let dist_mcp_wrist = magnitude(sub_points(lms[curlmcps[i]], p0));
        features.push(dist_tip_mcp / (dist_mcp_wrist + 1e-6));
    }

    // 3. Joint Angles (15)
    const chains = [
        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], 
        [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]
    ];
    for (let i = 0; i < 5; ++i) {
        for (let j = 1; j < 4; ++j) {
            let a = lms[chains[i][j-1]], b = lms[chains[i][j]], c = lms[chains[i][j+1]];
            let ba = sub_points(a, b), bc = sub_points(c, b);
            let m_ba = magnitude(ba), m_bc = magnitude(bc);
            features.push(dot_product(ba, bc) / (m_ba * m_bc + 1e-6));
        }
    }

    // 4. Face Context (23)
    // C++ checks: f.has_face && f.has_pose. Mirror that here.
    if (frame.face && frame.pose_anchors) {
        let face_h = magnitude(sub_points(frame.face.forehead, frame.face.chin));
        if (face_h < 0.01) face_h = 1.0;

        let idx_rel = { x: hand.wrist_pos.x + lms[8].x, y: hand.wrist_pos.y + lms[8].y, z: hand.wrist_pos.z + lms[8].z };
        let mid_rel = { x: hand.wrist_pos.x + lms[12].x, y: hand.wrist_pos.y + lms[12].y, z: hand.wrist_pos.z + lms[12].z };
        let tmb_rel = { x: hand.wrist_pos.x + lms[4].x, y: hand.wrist_pos.y + lms[4].y, z: hand.wrist_pos.z + lms[4].z };

        const l_ear = frame.pose_anchors.l_ear;
        const r_ear = frame.pose_anchors.r_ear;

        let anchors = [frame.face.forehead, frame.face.chin, frame.face.nose, frame.face.l_cheek, frame.face.r_cheek, l_ear, r_ear];
        
        anchors.forEach(a => features.push(magnitude(sub_points(idx_rel, a)) / face_h));
        anchors.forEach(a => features.push(magnitude(sub_points(mid_rel, a)) / face_h));
        anchors.forEach(a => features.push(magnitude(sub_points(tmb_rel, a)) / face_h));
        features.push(magnitude(sub_points(hand.wrist_pos, frame.face.forehead)) / face_h);
        features.push(magnitude(sub_points(hand.wrist_pos, frame.face.chin)) / face_h);
    } else {
        for (let i = 0; i < 23; i++) features.push(0.0);
    }

    // 5. Full Tip Matrix (10)
    for (let i = 0; i < 5; ++i) {
        for (let j = i + 1; j < 5; ++j) {
            features.push(magnitude(sub_points(lms[curltips[i]], lms[curltips[j]])) / hand_size);
        }
    }

    // 6. Thumb-Cross Matrix (4)
    const cross_pips = [6, 10, 14, 18];
    for (let i = 0; i < 4; ++i) {
        features.push(magnitude(sub_points(lms[4], lms[cross_pips[i]])) / hand_size);
    }

    // 7. Palm Orientation (3)
    let v1 = sub_points(lms[5], p0), v2 = sub_points(lms[17], p0);
    let normal = {
        x: v1.y * v2.z - v1.z * v2.y,
        y: v1.z * v2.x - v1.x * v2.z,
        z: v1.x * v2.y - v1.y * v2.x
    };
    let norm_mag = magnitude(normal);
    if (norm_mag > 1e-6) {
        features.push(normal.x / norm_mag, normal.y / norm_mag, normal.z / norm_mag);
    } else {
        features.push(0.0, 0.0, 0.0);
    }

    // 8. Orientation Highlighters (2)
    let idx_ratio = 0.0, mid_ratio = 0.0;
    const pairs = [[8, 5], [12, 9]];
    for (let i = 0; i < 2; ++i) {
        let dx = Math.abs(lms[pairs[i][0]].x - lms[pairs[i][1]].x);
        let dy = Math.abs(lms[pairs[i][0]].y - lms[pairs[i][1]].y);
        let ratio = dx / (dx + dy + 1e-6);
        features.push(ratio);
        if (i === 0) idx_ratio = ratio; else mid_ratio = ratio;
    }

    // 9. Feature Boosting (20)
    for (let k = 0; k < 10; ++k) {
        features.push(idx_ratio, mid_ratio);
    }

    return new Float32Array(features);
}

// ---------------------------------------------------------
// 2. Sparse Random Forest Evaluator
// ---------------------------------------------------------
export class SparseRandomForest {
    constructor(jsonData) {
        this.classes = jsonData.classes;
        this.n_classes = jsonData.n_classes;
        this.trees = jsonData.trees;
    }

    predict_proba(features) {
        const accum = new Float64Array(this.n_classes);
        for (const tree of this.trees) {
            let node = 0;
            while (tree.children_left[node] !== -1) {
                const feat = tree.feature[node];
                if (feat >= 0 && feat < features.length) {
                    if (features[feat] <= tree.threshold[node]) {
                        node = tree.children_left[node];
                    } else {
                        node = tree.children_right[node];
                    }
                } else {
                    break;
                }
            }
            // Add sparse values
            const valMap = tree.values[node];
            for (const classIdxStr in valMap) {
                accum[parseInt(classIdxStr)] += valMap[classIdxStr];
            }
        }
        
        for (let i = 0; i < this.n_classes; i++) {
            accum[i] /= this.trees.length;
        }
        return accum;
    }

    predict(features) {
        const probs = this.predict_proba(features);
        let maxIdx = 0;
        let maxVal = probs[0];
        for (let i = 1; i < this.n_classes; i++) {
            if (probs[i] > maxVal) {
                maxVal = probs[i];
                maxIdx = i;
            }
        }
        return { label: this.classes[maxIdx], confidence: maxVal };
    }
}

// ---------------------------------------------------------
// 3. Dynamic Time Warping (DTW & DDTW) Engine
// ---------------------------------------------------------
export class DtwEngine {
    static euclideanDistance(f1, f2) {
        let sum = 0.0;
        for (let i = 0; i < f1.length; i++) {
            let diff = f1[i] - f2[i];
            
            // Handshape Weighting
            if (i >= 0 && i < 80) diff *= 2.0;
            // Face Context Weighting
            if (i >= 80 && i <= 103) diff *= 4.0;
            // Palm Orientation Weighting
            if (i >= 77 && i <= 79) diff *= 3.0;

            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    static computeDerivatives(seq) {
        const derivatives = [];
        for (let i = 1; i < seq.length; i++) {
            const d = new Float32Array(seq[i].length);
            for (let k = 0; k < seq[i].length; k++) {
                // 8.0x velocity scaling
                d[k] = (seq[i][k] - seq[i-1][k]) * 8.0;
            }
            derivatives.push(d);
        }
        return derivatives;
    }

    static computeDTW(seq1, seq2, window = 0) {
        const n = seq1.length;
        const m = seq2.length;

        if (window <= 0) {
            window = Math.floor(Math.max(n, m) * 0.15);
            window = Math.max(window, Math.abs(n - m));
        }

        const dtw = Array.from({ length: n + 1 }, () => new Float32Array(m + 1).fill(999999.0));
        dtw[0][0] = 0.0;

        for (let i = 1; i <= n; i++) {
            const jStart = Math.max(1, i > window ? i - window : 1);
            const jEnd = Math.min(m, i + window);

            for (let j = jStart; j <= jEnd; j++) {
                const cost = DtwEngine.euclideanDistance(seq1[i-1], seq2[j-1]);
                dtw[i][j] = cost + Math.min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]);
            }
        }
        return dtw[n][m] / (n + m);
    }

    static computeDualScore(seq1, seq2, alpha = 0.5) {
        const shape_score = DtwEngine.computeDTW(seq1, seq2);
        
        const deriv1 = DtwEngine.computeDerivatives(seq1);
        const deriv2 = DtwEngine.computeDerivatives(seq2);
        const rhythm_score = DtwEngine.computeDTW(deriv1, deriv2);
        
        const duration_penalty = Math.abs(seq1.length - seq2.length) / Math.max(seq1.length, seq2.length) * 1.5;
        
        return (alpha * shape_score) + ((1.0 - alpha) * rhythm_score) + duration_penalty;
    }
}
