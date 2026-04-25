#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>

struct Point3D { float x, y, z; };

struct FaceStar {
    Point3D forehead, chin, nose, l_cheek, r_cheek, mouth, l_eye, r_eye;
};

struct PoseAnchors {
    Point3D l_ear, r_ear, l_shoulder, r_shoulder;
};

struct HandData {
    std::vector<Point3D> landmarks;
    Point3D wrist_pos;
};

struct Frame {
    std::vector<HandData> hands;
    FaceStar face;
    PoseAnchors pose;
    double timestamp;
    bool has_face = false;
    bool has_pose = false;
};

class DtwEngine {
public:
    static std::vector<std::vector<float>> extractFeatures(const std::vector<Frame>& sequence) {
        std::vector<std::vector<float>> features;
        if (sequence.empty()) return features;

        for (size_t i = 0; i < sequence.size(); ++i) {
            if (sequence[i].hands.empty()) continue; // Skip empty frames

            // Use Face Height as a "Ruler" for normalization (Scaling-Independence)
            float face_h = 1.0f;
            if (sequence[i].has_face) {
                float dx = sequence[i].face.forehead.x - sequence[i].face.chin.x;
                float dy = sequence[i].face.forehead.y - sequence[i].face.chin.y;
                float dz = sequence[i].face.forehead.z - sequence[i].face.chin.z;
                face_h = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (face_h < 0.01f) face_h = 1.0f; // Safety
            }

            std::vector<float> f;
            
            // ALWAYS extract exactly 2 hands (156 features total)
            for (size_t h_idx = 0; h_idx < 2; ++h_idx) {
                if (sequence[i].hands.size() > h_idx) {
                    const auto& hand = sequence[i].hands[h_idx];
                    
                    // --- NEW: Internal Hand Scaling ---
                    // Calculate distance from Wrist (0) to Middle MCP (9) to serve as a persistent "Hand Ruler"
                    float dx_h = hand.landmarks[9].x - hand.landmarks[0].x;
                    float dy_h = hand.landmarks[9].y - hand.landmarks[0].y;
                    float dz_h = hand.landmarks[9].z - hand.landmarks[0].z;
                    float hand_size = std::sqrt(dx_h*dx_h + dy_h*dy_h + dz_h*dz_h);
                    if (hand_size < 1e-6f) hand_size = 1.0f;

                    // 1. HANDSHAPE: 21 relative landmarks (NORMALIZED BY HAND SIZE)
                    for (size_t j = 0; j < 21; ++j) {
                        float weight = (j == 4 || j == 8 || j == 12 || j == 16 || j == 20) ? 5.0f : 1.0f; 
                        float sq_w = std::sqrt(weight);
                        f.push_back((hand.landmarks[j].x / hand_size) * sq_w);
                        f.push_back((hand.landmarks[j].y / hand_size) * sq_w);
                        f.push_back((hand.landmarks[j].z / hand_size) * sq_w);
                    }
                    
                    // 2. FINGER EXTENSION (NORMALIZED BY HAND SIZE)
                    int tips[] = {4, 8, 12, 16, 20};
                    for (int t : tips) {
                        float d = std::sqrt(std::pow(hand.landmarks[t].x, 2) + std::pow(hand.landmarks[t].y, 2) + std::pow(hand.landmarks[t].z, 2));
                        f.push_back((d / hand_size) * 6.0f);
                    }

                    // 3. FINGER SPREAD (NORMALIZED BY HAND SIZE)
                    for (int j = 0; j < 4; ++j) {
                        float dx = hand.landmarks[tips[j]].x - hand.landmarks[tips[j+1]].x;
                        float dy = hand.landmarks[tips[j]].y - hand.landmarks[tips[j+1]].y;
                        float dz = hand.landmarks[tips[j]].z - hand.landmarks[tips[j+1]].z;
                        f.push_back((std::sqrt(dx*dx + dy*dy + dz*dz) / hand_size) * 6.0f);
                    } // closing brace for the for loop!
                    
                    // 4. NOSE-RELATIVE POSITION (Directionality)
                    // Instead of trajectory from start, we just use absolute position relative to the nose.
                    f.push_back((hand.wrist_pos.x / face_h) * 20.0f);
                    f.push_back((hand.wrist_pos.y / face_h) * 20.0f);

                    // 5. FACE DISTANCES (16 Semantic Probes normalized by Face Height)
                    if (sequence[i].has_face && sequence[i].has_pose) {
                        auto& face = sequence[i].face;
                        auto& pIdx = sequence[i].pose; // Pose anchors for Ears
                        
                        Point3D anchors[] = { face.forehead, face.chin, face.nose, face.l_cheek, face.r_cheek, pIdx.l_ear, pIdx.r_ear };
                        
                        Point3D idx_rel_nose = { hand.wrist_pos.x + hand.landmarks[8].x, hand.wrist_pos.y + hand.landmarks[8].y, hand.wrist_pos.z + hand.landmarks[8].z };
                        Point3D mid_rel_nose = { hand.wrist_pos.x + hand.landmarks[12].x, hand.wrist_pos.y + hand.landmarks[12].y, hand.wrist_pos.z + hand.landmarks[12].z };
                        Point3D tmb_rel_nose = { hand.wrist_pos.x + hand.landmarks[4].x, hand.wrist_pos.y + hand.landmarks[4].y, hand.wrist_pos.z + hand.landmarks[4].z };

                        auto dist_norm = [&](Point3D a, Point3D b) {
                            return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2)) / face_h;
                        };

                        // 1. Index Proximity (7)
                        for (const auto& a : anchors) f.push_back(dist_norm(idx_rel_nose, a) * 8.0f);
                        // 2. Middle Proximity (7)
                        for (const auto& a : anchors) f.push_back(dist_norm(mid_rel_nose, a) * 8.0f);
                        // 3. Thumb Proximity (7)
                        for (const auto& a : anchors) f.push_back(dist_norm(tmb_rel_nose, a) * 25.0f); // Massive boost to Thumb Position!
                        // 4. Wrist Proximity (2)
                        f.push_back(dist_norm(hand.wrist_pos, face.forehead) * 8.0f);
                        f.push_back(dist_norm(hand.wrist_pos, face.chin) * 8.0f);
                    } else {
                        for (int k = 0; k < 23; ++k) f.push_back(0.0f);
                    }

                    // --- NEW: 6. JOINT ANGLES (15 Features) ---
                    // Scale and rotation invariant. Eliminates orientation errors.
                    int chains[5][5] = {
                        {0, 1, 2, 3, 4}, {0, 5, 6, 7, 8}, {0, 9, 10, 11, 12}, 
                        {0, 13, 14, 15, 16}, {0, 17, 18, 19, 20}
                    };
                    for (int c = 0; c < 5; ++c) {
                        for (int j = 1; j < 4; ++j) {
                            Point3D a = hand.landmarks[chains[c][j-1]];
                            Point3D b = hand.landmarks[chains[c][j]];
                            Point3D c_pt = hand.landmarks[chains[c][j+1]];
                            Point3D ba = {a.x - b.x, a.y - b.y, a.z - b.z};
                            Point3D bc = {c_pt.x - b.x, c_pt.y - b.y, c_pt.z - b.z};
                            float m_ba = std::sqrt(ba.x*ba.x + ba.y*ba.y + ba.z*ba.z);
                            float m_bc = std::sqrt(bc.x*bc.x + bc.y*bc.y + bc.z*bc.z);
                            float dot = (ba.x*bc.x + ba.y*bc.y + ba.z*bc.z) / (m_ba * m_bc + 1e-6f);
                            f.push_back(dot * 4.0f); // Boost angle importance 
                        }
                    }

                    // --- NEW: 7. PALM NORMAL VECTOR (3 Features) ---
                    // Cross product of Wrist->Index MCP and Wrist->Pinky MCP. Tells us where the palm is pointing.
                    Point3D v1 = {hand.landmarks[5].x - hand.landmarks[0].x, 
                                  hand.landmarks[5].y - hand.landmarks[0].y, 
                                  hand.landmarks[5].z - hand.landmarks[0].z};
                    Point3D v2 = {hand.landmarks[17].x - hand.landmarks[0].x, 
                                  hand.landmarks[17].y - hand.landmarks[0].y, 
                                  hand.landmarks[17].z - hand.landmarks[0].z};
                    Point3D norm = {v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x};
                    float mag_norm = std::sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z) + 1e-6f;
                    f.push_back((norm.x / mag_norm) * 5.0f);
                    f.push_back((norm.y / mag_norm) * 5.0f);
                    f.push_back((norm.z / mag_norm) * 5.0f);

                } else {
                    // PADDING: 115 features for a missing hand (63sh + 5ex + 4sp + 2tr + 23fc + 15ang + 3pm)
                    for (int k = 0; k < 115; ++k) f.push_back(0.0f);
                }
            }

            // 6. INTER-HAND PROXIMITY (4 Features) - The "Tapping" Detector
            // Symmetrical distance between hands. Extreme weight because this DEFINES tapping vs intact.
            if (sequence[i].hands.size() >= 2) {
                const auto& h1 = sequence[i].hands[0];
                const auto& h2 = sequence[i].hands[1];
                
                auto dist3d = [&](Point3D a, Point3D b) {
                    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2));
                };

                // Wrists
                f.push_back((dist3d(h1.wrist_pos, h2.wrist_pos) / face_h) * 50.0f);
                
                // Index Tips
                Point3D t1_idx = {h1.wrist_pos.x + h1.landmarks[8].x, h1.wrist_pos.y + h1.landmarks[8].y, h1.wrist_pos.z + h1.landmarks[8].z};
                Point3D t2_idx = {h2.wrist_pos.x + h2.landmarks[8].x, h2.wrist_pos.y + h2.landmarks[8].y, h2.wrist_pos.z + h2.landmarks[8].z};
                f.push_back((dist3d(t1_idx, t2_idx) / face_h) * 50.0f);
                
                // Middle Tips
                Point3D t1_mid = {h1.wrist_pos.x + h1.landmarks[12].x, h1.wrist_pos.y + h1.landmarks[12].y, h1.wrist_pos.z + h1.landmarks[12].z};
                Point3D t2_mid = {h2.wrist_pos.x + h2.landmarks[12].x, h2.wrist_pos.y + h2.landmarks[12].y, h2.wrist_pos.z + h2.landmarks[12].z};
                f.push_back((dist3d(t1_mid, t2_mid) / face_h) * 50.0f);

                // Thumb Tips
                Point3D t1_tmb = {h1.wrist_pos.x + h1.landmarks[4].x, h1.wrist_pos.y + h1.landmarks[4].y, h1.wrist_pos.z + h1.landmarks[4].z};
                Point3D t2_tmb = {h2.wrist_pos.x + h2.landmarks[4].x, h2.wrist_pos.y + h2.landmarks[4].y, h2.wrist_pos.z + h2.landmarks[4].z};
                f.push_back((dist3d(t1_tmb, t2_tmb) / face_h) * 50.0f);

                // (Padding to keep feature vector consistent: 14 features total allocated)
                while (f.size() < (230 + 14)) f.push_back(0.0f); 
            } else {
                for (int k = 0; k < 14; ++k) f.push_back(0.0f);
            }

            features.push_back(f);
        }
        return features;
    }

    static float euclideanDistance(const std::vector<float>& f1, const std::vector<float>& f2) {
        if (f1.size() != f2.size()) return 999.0f; // High penalty for hand-count mismatch
        float sum = 0.0f;
        for (size_t i = 0; i < f1.size(); ++i) {
            float diff = f1[i] - f2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // ========== UPGRADED: DDTW - Raw Energy Logic ==========
    // We now use RAW Velocity differences without normalization.
    // This allows the engine to distinguish between 'Forceful Taps' and 'Gentle Rocks'
    // because the mathematical 'Power' of the movement is preserved.
    static std::vector<std::vector<float>> computeDerivatives(const std::vector<std::vector<float>>& seq) {
        std::vector<std::vector<float>> derivatives;
        if (seq.size() < 2) return derivatives;

        for (size_t i = 1; i < seq.size(); ++i) {
            std::vector<float> d;
            size_t dim = std::min(seq[i].size(), seq[i-1].size());
            for (size_t k = 0; k < dim; ++k) {
                float val = (seq[i][k] - seq[i-1][k]) * 10.0f; // Raw Velocity Scaled
                d.push_back(val);
            }
            derivatives.push_back(d);
        }
        return derivatives;
    }

    // ========== UPGRADED: Standard DTW with Sakoe-Chiba Warping Window ==========
    // The "window" parameter prevents unrealistic time-stretching.
    // Set window = 0 for unlimited warping (original behavior).
    static float computeDTW(const std::vector<std::vector<float>>& seq1, const std::vector<std::vector<float>>& seq2, int window = 0) {
        if (seq1.empty() || seq2.empty()) return 999999.0f;
        
        size_t n = seq1.size(), m = seq2.size();
        
        // Auto-calculate window if not specified (15% of the longer sequence)
        if (window <= 0) {
            window = static_cast<int>(std::max(n, m) * 0.15);
            window = std::max(window, (int)std::abs((int)n - (int)m)); // Must be at least the length difference
        }

        std::vector<std::vector<float>> dtw(n + 1, std::vector<float>(m + 1, 999999.0f));
        dtw[0][0] = 0.0f;
        
        for (size_t i = 1; i <= n; ++i) {
            // Sakoe-Chiba: Only search within the window band around the diagonal
            size_t j_start = std::max((size_t)1, i > (size_t)window ? i - window : 1);
            size_t j_end   = std::min(m, i + window);

            for (size_t j = j_start; j <= j_end; ++j) {
                float cost = euclideanDistance(seq1[i-1], seq2[j-1]);
                dtw[i][j] = cost + std::min({dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]});
            }
        }
        
        // Return normalized path cost (average distance per frame step)
        return dtw[n][m] / (n + m); 
    }

    // ========== NEW: Dual-Score Fusion (DTW + DDTW) ==========
    // Combines shape matching (DTW) with rhythm matching (DDTW) for maximum confidence.
    // alpha controls the weight: 0.5 = equal, higher = more shape, lower = more rhythm.
    static float computeDualScore(const std::vector<std::vector<float>>& seq1, const std::vector<std::vector<float>>& seq2, float alpha = 0.5f) {
        // Score A: Standard DTW (Shape)
        float shape_score = computeDTW(seq1, seq2);

        // Score B: Derivative DTW (Rhythm/Velocity)
        auto deriv1 = computeDerivatives(seq1);
        auto deriv2 = computeDerivatives(seq2);
        float rhythm_score = computeDTW(deriv1, deriv2);

        // Weighted fusion
        float final_score = (alpha * shape_score) + ((1.0f - alpha) * rhythm_score);
        return final_score;
    }
};
