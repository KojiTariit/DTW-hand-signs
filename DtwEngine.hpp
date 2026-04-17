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

        for (size_t i = 1; i < sequence.size(); ++i) {
            std::vector<float> f;
            
            // For each hand present in the frame, extract its signature
            for (const auto& hand : sequence[i].hands) {
                // 1. HANDSHAPE: 21 relative landmarks (Weighted)
                for (size_t j = 0; j < 21; ++j) {
                    float weight = 1.0f;
                    if (j == 4 || j == 8 || j == 12 || j == 16 || j == 20) weight = 5.0f; // Boosted from 3.0
                    float sq_w = std::sqrt(weight);
                    f.push_back(hand.landmarks[j].x * sq_w);
                    f.push_back(hand.landmarks[j].y * sq_w);
                    f.push_back(hand.landmarks[j].z * sq_w);
                }
                
                // 2. FINGER EXTENSION
                int tips[] = {4, 8, 12, 16, 20};
                for (int t : tips) {
                    float d = std::sqrt(std::pow(hand.landmarks[t].x, 2) + 
                                       std::pow(hand.landmarks[t].y, 2) + 
                                       std::pow(hand.landmarks[t].z, 2));
                    f.push_back(d * 6.0f); // Boosted from 4.0
                }

                // 3. FINGER SPREAD
                for (int j = 0; j < 4; ++j) {
                    float dx = hand.landmarks[tips[j]].x - hand.landmarks[tips[j+1]].x;
                    float dy = hand.landmarks[tips[j]].y - hand.landmarks[tips[j+1]].y;
                    float dz = hand.landmarks[tips[j]].z - hand.landmarks[tips[j+1]].z;
                    float d = std::sqrt(dx*dx + dy*dy + dz*dz);
                    f.push_back(d * 6.0f); // Boosted from 4.0
                }
                
                // 4. TRAJECTORY (Relative to start of capture)
                // Note: Indexing into sequence[0].hands[0] assumes hand indices stay consistent
                if (!sequence[0].hands.empty()) {
                   float dx_traj = hand.wrist_pos.x - sequence[0].hands[0].wrist_pos.x;
                   float dy_traj = hand.wrist_pos.y - sequence[0].hands[0].wrist_pos.y;
                   f.push_back(dx_traj * 7.0f);
                   f.push_back(dy_traj * 7.0f);
                } else {
                   f.push_back(0); f.push_back(0);
                }
            }
            // If it's a 2-hand sign, the feature vector will be 2x longer automatically.
            // DTW will naturally fail a 1-hand live sign against a 2-hand template because the vector lengths won't match!
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

    // ========== NEW: DDTW - Derivative (Velocity) Feature Extraction ==========
    // Takes a sequence of feature vectors and returns the "velocity" between consecutive frames.
    // Instead of "where is the finger?", this tells us "how fast is the finger moving?"
    static std::vector<std::vector<float>> computeDerivatives(const std::vector<std::vector<float>>& seq) {
        std::vector<std::vector<float>> derivatives;
        if (seq.size() < 2) return derivatives;

        for (size_t i = 1; i < seq.size(); ++i) {
            std::vector<float> d;
            size_t dim = std::min(seq[i].size(), seq[i-1].size());
            for (size_t k = 0; k < dim; ++k) {
                d.push_back(seq[i][k] - seq[i-1][k]); // Velocity = Current - Previous
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
