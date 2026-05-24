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
    bool is_present = false;
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

        for (const auto& f : sequence) {
            // Find first valid hand (Sniper Core+ only supports one hand for movement)
            const HandData* target_hand = nullptr;
            for (const auto& h : f.hands) {
                if (h.is_present && h.landmarks.size() >= 21) {
                    target_hand = &h;
                    break;
                }
            }
            if (!target_hand) continue;

            const auto& hand = *target_hand;
            const auto& lms = hand.landmarks;
            
            float dx_sz = lms[9].x - lms[0].x;
            float dy_sz = lms[9].y - lms[0].y;
            float dz_sz = lms[9].z - lms[0].z;
            float hand_size = std::sqrt(dx_sz*dx_sz + dy_sz*dy_sz + dz_sz*dz_sz);
            if (hand_size < 1e-6f) hand_size = 1.0f;

            std::vector<float> feat;
            
            // 1. Wrist-Relative XYZ Coordinates (60) - Solves Orientation (H vs R)
            for (int i = 1; i < 21; ++i) {
                feat.push_back((lms[i].x - lms[0].x) / hand_size);
                feat.push_back((lms[i].y - lms[0].y) / hand_size);
                feat.push_back((lms[i].z - lms[0].z) / hand_size);
            }

            // 2. Finger Extension Ratios (5)
            int tips[] = {4, 8, 12, 16, 20};
            int mcps[] = {2, 5, 9, 13, 17};
            for (int i = 0; i < 5; ++i) {
                float d_tip = std::sqrt(std::pow(lms[tips[i]].x - lms[mcps[i]].x, 2) + std::pow(lms[tips[i]].y - lms[mcps[i]].y, 2) + std::pow(lms[tips[i]].z - lms[mcps[i]].z, 2));
                float d_base = std::sqrt(std::pow(lms[mcps[i]].x - lms[0].x, 2) + std::pow(lms[mcps[i]].y - lms[0].y, 2) + std::pow(lms[mcps[i]].z - lms[0].z, 2));
                feat.push_back(d_tip / (d_base + 1e-6f));
            }

            // 3. Joint Angles (15)
            int chains[5][5] = {{0,1,2,3,4}, {0,5,6,7,8}, {0,9,10,11,12}, {0,13,14,15,16}, {0,17,18,19,20}};
            for (int i = 0; i < 5; ++i) {
                for (int j = 1; j < 4; ++j) {
                    Point3D a = lms[chains[i][j-1]], b = lms[chains[i][j]], c = lms[chains[i][j+1]];
                    Point3D ba = {a.x-b.x, a.y-b.y, a.z-b.z}, bc = {c.x-b.x, c.y-b.y, c.z-b.z};
                    float mba = std::sqrt(ba.x*ba.x+ba.y*ba.y+ba.z*ba.z), mbc = std::sqrt(bc.x*bc.x+bc.y*bc.y+bc.z*bc.z);
                    feat.push_back((ba.x*bc.x+ba.y*bc.y+ba.z*bc.z) / (mba*mbc + 1e-6f));
                }
            }

            // 4. Face Context (23) - WITH 4.0X SPATIAL WEIGHTING
            if (f.has_face) {
                float dx_f = f.face.forehead.x - f.face.chin.x;
                float dy_f = f.face.forehead.y - f.face.chin.y;
                float dz_f = f.face.forehead.z - f.face.chin.z;
                float face_h = std::sqrt(dx_f*dx_f + dy_f*dy_f + dz_f*dz_f);
                if (face_h < 0.01f) face_h = 1.0f;

                Point3D i_rel = {hand.wrist_pos.x + lms[8].x, hand.wrist_pos.y + lms[8].y, hand.wrist_pos.z + lms[8].z};
                Point3D m_rel = {hand.wrist_pos.x + lms[12].x, hand.wrist_pos.y + lms[12].y, hand.wrist_pos.z + lms[12].z};
                Point3D t_rel = {hand.wrist_pos.x + lms[4].x, hand.wrist_pos.y + lms[4].y, hand.wrist_pos.z + lms[4].z};
                
                // Use face anchors. Pose anchors (ears) will be {0,0,0} if missing but the weight is on face.
                Point3D anchors[] = {f.face.forehead, f.face.chin, f.face.nose, f.face.mouth, f.face.l_cheek, f.face.r_cheek, f.pose.l_ear};

                auto dnorm = [&](Point3D a, Point3D b) {
                    return std::sqrt(std::pow(a.x-b.x,2)+std::pow(a.y-b.y,2)+std::pow(a.z-b.z,2)) / face_h;
                };

                for (auto& a : anchors) feat.push_back(dnorm(i_rel, a));
                for (auto& a : anchors) feat.push_back(dnorm(m_rel, a));
                for (auto& a : anchors) feat.push_back(dnorm(t_rel, a));
                feat.push_back(dnorm(hand.wrist_pos, f.face.forehead));
                feat.push_back(dnorm(hand.wrist_pos, f.face.chin));
            } else {
                for (int k = 0; k < 23; ++k) feat.push_back(0.0f);
            }

            // 5. Full Tip Matrix (10)
            for (int i = 0; i < 5; ++i) {
                for (int j = i+1; j < 5; ++j) {
                    float dx = lms[tips[i]].x - lms[tips[j]].x, dy = lms[tips[i]].y - lms[tips[j]].y, dz = lms[tips[i]].z - lms[tips[j]].z;
                    feat.push_back(std::sqrt(dx*dx+dy*dy+dz*dz) / hand_size);
                }
            }

            // 6. Thumb-Cross Matrix (4)
            int pips[] = {6, 10, 14, 18};
            for (int p : pips) {
                float dx = lms[4].x - lms[p].x, dy = lms[4].y - lms[p].y, dz = lms[4].z - lms[p].z;
                feat.push_back(std::sqrt(dx*dx+dy*dy+dz*dz) / hand_size);
            }

            // 7. Palm Orientation (3 Features: Palm Normal X, Y, Z)
            Point3D v1_n = {lms[5].x - lms[0].x, lms[5].y - lms[0].y, lms[5].z - lms[0].z};
            Point3D v2_n = {lms[17].x - lms[0].x, lms[17].y - lms[0].y, lms[17].z - lms[0].z};
            Point3D normal = {
                v1_n.y*v2_n.z - v1_n.z*v2_n.y,
                v1_n.z*v2_n.x - v1_n.x*v2_n.z,
                v1_n.x*v2_n.y - v1_n.y*v2_n.x
            };
            float norm_mag = std::sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
            if (norm_mag > 1e-6f) {
                feat.push_back(normal.x / norm_mag);
                feat.push_back(normal.y / norm_mag);
                feat.push_back(normal.z / norm_mag);
            } else {
                feat.push_back(0.0f); feat.push_back(0.0f); feat.push_back(0.0f);
            }

            features.push_back(feat);
        }
        return features;
    }

    static float euclideanDistance(const std::vector<float>& f1, const std::vector<float>& f2) {
        if (f1.size() != f2.size()) return 999.0f;
        float sum = 0.0f;
        for (size_t i = 0; i < f1.size(); ++i) {
            float diff = f1[i] - f2[i];
            
            // --- HANDSHAPE WEIGHTING ---
            if (i >= 0 && i < 80) { // XYZ (60) + Curl (5) + Angles (15)
                diff *= 2.0f;
            }

            // --- SPATIAL STAR WEIGHTING ---
            if (i >= 80 && i <= 103) { // Face (23)
                diff *= 4.0f; 
            }

            // --- PALM ORIENTATION WEIGHTING (NEW) ---
            // Indices 77, 78, 79
            if (i >= 77 && i <= 79) {
                diff *= 3.0f; 
            }
            
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
                // RHYTHM BOOST: 8.0x scaling to emphasize movement velocity
                float val = (seq[i][k] - seq[i-1][k]) * 8.0f; 
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

        // Score C: Duration Penalty
        float n = (float)seq1.size();
        float m = (float)seq2.size();
        float duration_penalty = std::abs(n - m) / std::max(n, m) * 1.5f;

        // Weighted fusion
        float final_score = (alpha * shape_score) + ((1.0f - alpha) * rhythm_score) + duration_penalty;
        return final_score;
    }
};
