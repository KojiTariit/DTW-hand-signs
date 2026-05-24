#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include "json.hpp"
#include "SignDatabase.hpp"
#include "DtwEngine.hpp"
#include "ForestClassifier.hpp"

using json = nlohmann::json;
float current_ml_power = 0.75f; // Global state for live tuning

ForestClassifier static_classifier;
ForestClassifier dynamic_classifier;

// --- 1. MATH HELPERS ---
float magnitude(const Point3D& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float dot_product(const Point3D& v1, const Point3D& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

Point3D sub_points(const Point3D& a, const Point3D& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

float dist3D(Point3D a, Point3D b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2));
}

// --- 2. X-RAY MACHINE LEARNING EXTRACTOR ---
// Extracts exactly what m2cgen expects (80 features)
std::vector<float> extract_ml_features(const Frame& f) {
    std::vector<float> features;
    if (f.hands.empty()) return features;
    
    const HandData* target_hand = nullptr;
    for (const auto& h : f.hands) {
        if (h.is_present && h.landmarks.size() >= 21) {
            target_hand = &h;
            break;
        }
    }
    if (!target_hand) return features;
    
    const auto& hand = *target_hand;
    const auto& lms = hand.landmarks;
    
    Point3D p0 = lms[0];
    Point3D p9 = lms[9];
    float hand_size = magnitude(sub_points(p9, p0));
    if (hand_size < 1e-6f) hand_size = 1.0f;

    // 1. Wrist-Relative XYZ Coordinates (60 Features) - Solves Orientation (H vs R)
    for (int i = 1; i < 21; ++i) {
        Point3D d = sub_points(lms[i], p0);
        features.push_back(d.x / hand_size);
        features.push_back(d.y / hand_size);
        features.push_back(d.z / hand_size);
    }

    // 2. Finger Extension Ratios (5 Features) - Curl Detection
    int curltips[] = {4, 8, 12, 16, 20};
    int curlmcps[] = {2, 5, 9, 13, 17};
    for (int i = 0; i < 5; ++i) {
        float dist_tip_mcp = magnitude(sub_points(lms[curltips[i]], lms[curlmcps[i]]));
        float dist_mcp_wrist = magnitude(sub_points(lms[curlmcps[i]], p0));
        features.push_back(dist_tip_mcp / (dist_mcp_wrist + 1e-6f));
    }

    // 3. Joint Angles (15 Features)
    int chains[5][5] = {
        {0, 1, 2, 3, 4}, {0, 5, 6, 7, 8}, {0, 9, 10, 11, 12}, 
        {0, 13, 14, 15, 16}, {0, 17, 18, 19, 20}
    };
    for (int i = 0; i < 5; ++i) {
        for (int j = 1; j < 4; ++j) {
            Point3D a = lms[chains[i][j-1]];
            Point3D b = lms[chains[i][j]];
            Point3D c = lms[chains[i][j+1]];
            Point3D ba = sub_points(a, b);
            Point3D bc = sub_points(c, b);
            float m_ba = magnitude(ba);
            float m_bc = magnitude(bc);
            features.push_back(dot_product(ba, bc) / (m_ba * m_bc + 1e-6f));
        }
    }

    // 4. Face Context (23 Semantic Probes) - Normalized by Face Height
    if (f.has_face && f.has_pose) {
        float face_h = magnitude(sub_points(f.face.forehead, f.face.chin));
        if (face_h < 0.01f) face_h = 1.0f;
        
        Point3D idx_rel_nose = {hand.wrist_pos.x + lms[8].x, hand.wrist_pos.y + lms[8].y, hand.wrist_pos.z + lms[8].z};
        Point3D mid_rel_nose = {hand.wrist_pos.x + lms[12].x, hand.wrist_pos.y + lms[12].y, hand.wrist_pos.z + lms[12].z};
        Point3D tmb_rel_nose = {hand.wrist_pos.x + lms[4].x, hand.wrist_pos.y + lms[4].y, hand.wrist_pos.z + lms[4].z};
        Point3D wst_rel_nose = hand.wrist_pos;
        
        Point3D anchors[] = { f.face.forehead, f.face.chin, f.face.nose, f.face.l_cheek, f.face.r_cheek, f.pose.l_ear, f.pose.r_ear };
        
        for (const auto& a : anchors) features.push_back(magnitude(sub_points(idx_rel_nose, a)) / face_h);
        for (const auto& a : anchors) features.push_back(magnitude(sub_points(mid_rel_nose, a)) / face_h);
        for (const auto& a : anchors) features.push_back(magnitude(sub_points(tmb_rel_nose, a)) / face_h);
        features.push_back(magnitude(sub_points(wst_rel_nose, f.face.forehead)) / face_h);
        features.push_back(magnitude(sub_points(wst_rel_nose, f.face.chin)) / face_h);
    } else {
        for (int i = 0; i < 23; ++i) features.push_back(0.0f);
    }

    // 5. Full Tip Matrix (10 Features) - Solves U vs V
    int tips[] = {4, 8, 12, 16, 20};
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            features.push_back(magnitude(sub_points(lms[tips[i]], lms[tips[j]])) / hand_size);
        }
    }

    // 6. Thumb-Cross Matrix (4 Features) - Solves A vs S (PIPs)
    int cross_pips[] = {6, 10, 14, 18};
    for (int i = 0; i < 4; ++i) {
        features.push_back(magnitude(sub_points(lms[4], lms[cross_pips[i]])) / hand_size);
    }

    // 7. Palm Orientation (3 Features: Normal X, Y, Z) - Solves Palm Up/Down
    Point3D v1 = sub_points(lms[5], lms[0]);
    Point3D v2 = sub_points(lms[17], lms[0]);
    Point3D normal = {
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    };
    float norm_mag = magnitude(normal);
    if (norm_mag > 1e-6f) {
        features.push_back(normal.x / norm_mag);
        features.push_back(normal.y / norm_mag);
        features.push_back(normal.z / norm_mag);
    } else {
        features.push_back(0.0f); features.push_back(0.0f); features.push_back(0.0f);
    }

    // 8. Orientation Highlighters (2 Features: Index and Middle X/Y Ratios)
    int highlighter_pairs[2][2] = {{8, 5}, {12, 9}};
    float idx_ratio = 0.0f, mid_ratio = 0.0f;
    for (int i = 0; i < 2; ++i) {
        float dx = std::abs(lms[highlighter_pairs[i][0]].x - lms[highlighter_pairs[i][1]].x);
        float dy = std::abs(lms[highlighter_pairs[i][0]].y - lms[highlighter_pairs[i][1]].y);
        float ratio = dx / (dx + dy + 1e-6f);
        features.push_back(ratio);
        if (i == 0) idx_ratio = ratio; else mid_ratio = ratio;
    }

    // 9. FEATURE BOOSTING: Duplicate highlighters 10x to force AI focus
    for (int k = 0; k < 10; ++k) {
        features.push_back(idx_ratio);
        features.push_back(mid_ratio);
    }

    return features;
}

#pragma comment(lib, "ws2_32.lib")

// --- 3. CLUSTER PRUNING BRAIN ---
struct ClusterModel {
    int n_clusters = 0;
    std::vector<std::vector<float>> centroids;
    std::vector<float> scaler_mean;
    std::vector<float> scaler_scale;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        try {
            json data = json::parse(f);
            n_clusters = data["n_clusters"];
            centroids = data["centroids"].get<std::vector<std::vector<float>>>();
            scaler_mean = data["scaler_mean"].get<std::vector<float>>();
            scaler_scale = data["scaler_scale"].get<std::vector<float>>();
            return true;
        } catch (...) { return false; }
    }

    std::vector<int> getTopClusters(const std::vector<float>& features, int k = 3) {
        if (features.size() != scaler_mean.size()) return {};
        
        std::vector<float> scaled(features.size());
        for (size_t i = 0; i < features.size(); ++i) {
            scaled[i] = (features[i] - scaler_mean[i]) / (scaler_scale[i] + 1e-6f);
        }

        std::vector<std::pair<int, float>> distances;
        for (int i = 0; i < n_clusters; ++i) {
            float dist = 0;
            for (size_t j = 0; j < scaled.size(); ++j) {
                dist += std::pow(scaled[j] - centroids[i][j], 2);
            }
            distances.push_back({i, std::sqrt(dist)});
        }

        std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

        std::vector<int> top;
        for (int i = 0; i < std::min(k, (int)distances.size()); ++i) {
            top.push_back(distances[i].first);
        }
        return top;
    }
};

int main() {
    std::cout << "--- SIGN RECOGNITION ENGINE V3.5 (REVOLUTION) ---" << std::endl;
    
    SignDatabase db;
    std::cout << "[SYSTEM] Initializing Template Database..." << std::endl;
    db.loadFromDirectory("templates", true);
    db.loadFromDirectory("templates_backup", false);
    db.loadFromDirectory("templateGundum", false);
    
    // Explicitly check for a root 'movement' folder if it exists
    if (std::filesystem::exists("movement")) {
        std::cout << "[SYSTEM] Found standalone movement folder. Merging..." << std::endl;
        db.loadFromDirectory("movement", false);
    }

    std::cout << "[SYSTEM] Initializing Machine Learning Forests..." << std::endl;
    if (!static_classifier.load("model_output/static_forest.json")) {
        std::cerr << "[WARNING] Failed to load static forest model!" << std::endl;
    }
    if (!dynamic_classifier.load("model_output/dynamic_forest.json")) {
        std::cerr << "[WARNING] Failed to load dynamic forest model!" << std::endl;
    }

    ClusterModel cluster_brain;
    bool cluster_enabled = cluster_brain.load("model_output/cluster_model.json");
    if (cluster_enabled) std::cout << "[REVOLUTION] Cluster Pruning ENABLED (Top-3 Probing)." << std::endl;
    else std::cout << "[WARNING] Cluster model not found. Running in Full-Scan mode." << std::endl;

    // 2. Setup UDP Server
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    SOCKET recvSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    sockaddr_in recvAddr;
    recvAddr.sin_family = AF_INET;
    recvAddr.sin_port = htons(5005);
    recvAddr.sin_addr.s_addr = INADDR_ANY;
    bind(recvSocket, (sockaddr*)&recvAddr, sizeof(recvAddr));

    std::cout << "Listening for Python Data on port 5005..." << std::endl;

    char buffer[16384];
    sockaddr_in senderAddr;
    int senderAddrSize = sizeof(senderAddr);
    std::vector<Frame> current_sign_buffer;
    std::string current_sentence = "";
    std::string last_prediction = "";

    // LATTICE: Stores a list of Top 3 candidates for every sign in the batch
    struct WordCandidates { std::vector<std::string> options; };
    std::vector<WordCandidates> sentence_lattice;

    std::cout << "\n======================================================\n";
    std::cout << "  CONTINUOUS LATTICE ENGINE ONLINE (V1.2)  \n";
    std::cout << "  - Just sign continuously. \n";
    std::cout << "  - Press 'F' to finish and see the full Lattice.\n";
    std::cout << "======================================================\n\n";

    while (true) {
        int bytesReceived = recvfrom(recvSocket, buffer, sizeof(buffer) - 1, 0, (sockaddr*)&senderAddr, &senderAddrSize);
        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0';
            std::string message(buffer);
            
            try {
                json data = json::parse(message);
                
                if (data.contains("ml_power")) {
                    current_ml_power = data["ml_power"];
                }
                
                if (data.contains("type") && data["type"] == "END_OF_SIGN") {
                    std::cout << "\n[SIGN COMPLETE] Processing " << current_sign_buffer.size() << " frames." << std::endl;
                    
                    if (current_sign_buffer.size() >= 5) {
                        WordCandidates wc;
                        // --- 3. ANALYZE MOVEMENT ---
                        size_t max_hands = 0;
                        float max_wrist_dist = 0.0f;
                        float max_shape_variance = 0.0f;
                        
                        std::vector<Point3D> start_wrist; 
                        std::vector<std::vector<Point3D>> start_shape; 
                        bool init_done = false;

                        for (const auto& f : current_sign_buffer) {
                            size_t frame_hands = 0;
                            for (const auto& h : f.hands) if (h.is_present) frame_hands++;
                            max_hands = std::max(max_hands, frame_hands);
                            
                            if (!init_done && frame_hands > 0) {
                                for (const auto& h : f.hands) {
                                    if (h.is_present) {
                                        start_wrist.push_back(h.wrist_pos);
                                        start_shape.push_back(h.landmarks);
                                    } else {
                                        start_wrist.push_back({-1000, -1000, -1000}); // Sentinel for missing hand
                                        start_shape.push_back({});
                                    }
                                }
                                init_done = true;
                            }

                            if (init_done) {
                                for (size_t i = 0; i < f.hands.size() && i < start_wrist.size(); ++i) {
                                    if (f.hands[i].is_present && start_wrist[i].x != -1000) {
                                        float dx = f.hands[i].wrist_pos.x - start_wrist[i].x;
                                        float dy = f.hands[i].wrist_pos.y - start_wrist[i].y;
                                        float dz = f.hands[i].wrist_pos.z - start_wrist[i].z;
                                        max_wrist_dist = std::max(max_wrist_dist, std::sqrt(dx*dx + dy*dy + dz*dz));

                                        for (size_t j = 0; j < f.hands[i].landmarks.size() && j < start_shape[i].size(); ++j) {
                                            float sx = f.hands[i].landmarks[j].x - start_shape[i][j].x;
                                            float sy = f.hands[i].landmarks[j].y - start_shape[i][j].y;
                                            float sz = f.hands[i].landmarks[j].z - start_shape[i][j].z;
                                            max_shape_variance = std::max(max_shape_variance, std::sqrt(sx*sx + sy*sy + sz*sz));
                                        }
                                    }
                                }
                            }
                        }

                        // --- 4. SMART ROUTING TREE ---
                        std::string winner = "None";
                        float TH_WRIST = 0.20f; 
                        float TH_SHAPE = 0.12f; 
                        bool is_dynamic = (current_sign_buffer.size() >= 12) && (max_wrist_dist > TH_WRIST || max_shape_variance > TH_SHAPE);

                        if (is_dynamic) {
                            std::cout << "  >> Route: DYNAMIC (DTW Processor w/ Spatial Pruning) <<" << std::endl;
                            
                            // NEW SPATIAL PRUNING: Frame Thresholding
                            int two_hand_frames = 0;
                            for (const auto& f : current_sign_buffer) {
                                size_t count = 0;
                                for (const auto& h : f.hands) if (h.is_present) count++;
                                if (count >= 2) two_hand_frames++;
                            }
                            
                            std::string target_cat = (two_hand_frames >= 5) ? "movement/2_hands" : "movement/single_hand";
                            std::cout << "  >> Filter: Searching [" << target_cat << "] folder only (2-hand frames: " << two_hand_frames << ")." << std::endl;

                            auto live_feat = DtwEngine::extractFeatures(current_sign_buffer);

                            // --- 3.5. ML SHAPE PRE-FILTER ---
                            std::cout << "  >> Filter: Generating ML Shape Shortlist..." << std::endl;
                            std::map<std::string, double> shape_votes;
                            for (size_t i = 0; i < current_sign_buffer.size(); ++i) {
                                // STRIDE OPTIMIZATION: Every 4th frame + heavy weight on "Money Frames"
                                if (i % 4 == 0 || i == 10 || i == 15) {
                                    auto ml_feat = extract_ml_features(current_sign_buffer[i]);
                                    if (ml_feat.size() >= 120) {
                                        std::vector<float> dyn_feat(ml_feat.begin(), ml_feat.begin() + 120);
                                        auto probs = dynamic_classifier.predict_proba(dyn_feat);
                                        auto classes = dynamic_classifier.get_classes();
                                        double weight = (i == 10 || i == 15) ? 5.0 : 1.0;
                                        for (size_t k = 0; k < probs.size(); ++k) {
                                            shape_votes[classes[k]] += probs[k] * weight;
                                        }
                                    }
                                }
                            }
                            
                            // --- NEW: CLUSTER PRUNING (Narrowing the search) ---
                            std::vector<int> allowed_clusters;
                            if (cluster_enabled) {
                                 std::vector<float> avg_feat(120, 0.0f);
                                 int count = 0;
                                 size_t start_f = (live_feat.size() > 10) ? 5 : 0;
                                 for (size_t i = start_f; i < live_feat.size(); ++i) {
                                     for (int k = 0; k < 120; ++k) avg_feat[k] += live_feat[i][k];
                                     count++;
                                 }
                                 if (count > 0) {
                                     for (int k = 0; k < 120; ++k) avg_feat[k] /= count;
                                 }
                                 
                                 allowed_clusters = cluster_brain.getTopClusters(avg_feat, 3);
                            }

                            // Find Top 10 ML Candidates for Fusion
                            std::vector<std::pair<std::string, double>> sorted_votes(shape_votes.begin(), shape_votes.end());
                            std::sort(sorted_votes.begin(), sorted_votes.end(), [](const auto& a, const auto& b) {
                                return a.second > b.second;
                            });
                            
                            double total_votes = 0;
                            for (const auto& v : sorted_votes) total_votes += v.second;
                            if (total_votes < 0.1) total_votes = 1.0;

                            // Step 2: TRI-FACTOR FUSION (ML + CLUSTER + DTW)
                            struct FusionCandidate { std::string name; float fused_score; float dtw_dist; float ml_bonus; float cluster_bonus; std::string folder; };
                            std::vector<FusionCandidate> candidates;

                            size_t top_n = std::min((size_t)5, sorted_votes.size()); // Top 5 only
                            for (size_t i = 0; i < top_n; ++i) {
                                std::string name = sorted_votes[i].first;
                                double confidence = sorted_votes[i].second / total_votes;
                                if (confidence < 0.001) continue;

                                // 1. Calculate Cluster Proximity Bonus
                                float cluster_bonus = 0.0f;
                                if (cluster_enabled && !allowed_clusters.empty() && db.file_to_cluster.count(name)) {
                                     int sign_cluster = db.file_to_cluster.at(name);
                                     if (allowed_clusters.size() > 0 && sign_cluster == allowed_clusters[0]) cluster_bonus = 0.30f;
                                     else if (allowed_clusters.size() > 1 && sign_cluster == allowed_clusters[1]) cluster_bonus = 0.24f;
                                     else if (allowed_clusters.size() > 2 && sign_cluster == allowed_clusters[2]) cluster_bonus = 0.19f;
                                }

                                // 2. Calculate ML Confidence Bonus (Exponential)
                                float ml_bonus = (float)std::sqrt(confidence) * 1.50f;

                                // 3. Cap Total Bonus at 95%
                                float total_bonus = std::min(0.95f, ml_bonus + cluster_bonus);

                                float dtw_dist = 999.0f;
                                bool found = false;
                                std::string actual_folder = "";
                                std::vector<std::string> all_folders = {"movement/single_hand", "movement/2_hands"};
                                for (const std::string& folder : all_folders) {
                                    if (db.categorized_templates.count(folder) && db.categorized_templates.at(folder).count(name)) {
                                        auto template_feat = db.categorized_templates.at(folder).at(name);
                                        dtw_dist = DtwEngine::computeDualScore(live_feat, template_feat, 0.4f);
                                        found = true;
                                        actual_folder = folder;
                                        break;
                                    }
                                }

                                if (found) {
                                     float movement_score = dtw_dist;
                                     float ai_score = (1.0f - (float)confidence) * 50.0f - (cluster_bonus * 30.0f);
                                     float fused = ((1.0f - current_ml_power) * movement_score) + (current_ml_power * ai_score);
                                     candidates.push_back({name, fused, dtw_dist, ml_bonus, cluster_bonus, actual_folder});
                                }
                            }
                             
                            // Sort by Fused Score (Lowest wins)
                            std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                                return a.fused_score < b.fused_score;
                            });

                            if (!candidates.empty()) {
                                std::cout << "  >> [RAW SCOREBOARD - FULL PICTURE]:" << std::endl;
                                for(int i=0; i<std::min((int)3, (int)candidates.size()); ++i) {
                                    std::string hand_tag = (candidates[i].folder == "movement/2_hands") ? "[2H]" : "[1H]";
                                    std::cout << "     " << (i+1) << ". " << candidates[i].name << " " << hand_tag 
                                              << " | Fused: " << candidates[i].fused_score 
                                              << " (DTW: " << candidates[i].dtw_dist << ")" << std::endl;
                                }
                                
                                // PRUNE & SHOW TOP 3 MATCHING HAND COUNT
                                std::vector<FusionCandidate> pruned_list;
                                for (const auto& cand : candidates) {
                                    if (cand.folder == target_cat) {
                                        pruned_list.push_back(cand);
                                    }
                                }

                                if (!pruned_list.empty()) {
                                    std::cout << "  >> [PRUNED SCOREBOARD - " << (target_cat == "movement/2_hands" ? "2 HANDS" : "1 HAND") << " ONLY]:" << std::endl;
                                    for(int i=0; i<std::min((int)3, (int)pruned_list.size()); ++i) {
                                        std::cout << "     " << (i+1) << ". " << pruned_list[i].name 
                                                  << " | Fused: " << pruned_list[i].fused_score 
                                                  << " (DTW: " << pruned_list[i].dtw_dist << ")" << std::endl;
                                    }
                                    winner = pruned_list[0].name;
                                } else {
                                    winner = candidates[0].name; 
                                    std::cout << "  !! WARNING: No " << target_cat << " signs found in ML shortlist. Using raw winner." << std::endl;
                                }

                                // Add Top Options to Lattice
                                const auto& active_list = (!pruned_list.empty()) ? pruned_list : candidates;
                                for(int i=0; i<std::min((int)3, (int)active_list.size()); ++i) {
                                    wc.options.push_back(active_list[i].name);
                                }
                                if (!wc.options.empty()) sentence_lattice.push_back(wc);
                            } else {
                                if (!sorted_votes.empty()) {
                                    winner = sorted_votes[0].first;
                                    std::cout << "  !! WARNING: No templates found for shortlist candidates. Falling back to top ML class: " << winner << std::endl;
                                    for (size_t i = 0; i < std::min((size_t)3, sorted_votes.size()); ++i) {
                                        wc.options.push_back(sorted_votes[i].first);
                                    }
                                    if (!wc.options.empty()) sentence_lattice.push_back(wc);
                                }
                            }

                        } else {
                            std::cout << "  >> Route: STATIC (ML Processor) <<" << std::endl;
                            const Frame& mid_f = current_sign_buffer[current_sign_buffer.size() / 2];
                            if (!mid_f.hands.empty()) {
                                std::vector<float> ml_features = extract_ml_features(mid_f);
                                if (ml_features.size() == 142) {
                                    winner = static_classifier.predict(ml_features);
                                    
                                    // --- THE SPATIAL GUARD ---
                                    float idx_r = ml_features[120];
                                    float mid_r = ml_features[121];
                                    float avg_angle = (idx_r + mid_r) / 2.0f;

                                    if (avg_angle > 0.65f) { // Sideways Hand
                                        if (winner == "R" || winner == "U" || winner == "V" || winner == "I") {
                                            winner = "H"; // Force H
                                        }
                                    } else if (avg_angle < 0.35f) { // Vertical Hand
                                        if (winner == "H") {
                                            winner = "U"; // Force U
                                        }
                                    }

                                    // Add single winner to Lattice (Static Route)
                                    if (winner != "NONE" && winner != "" && winner != "None") {
                                        wc.options.push_back(winner);
                                        sentence_lattice.push_back(wc);
                                    }
                                }
                            }
                        }
                        
                        // Lattice is now handled inside each route above

                        if (winner != "" && winner != last_prediction && winner != "None") {
                            if (!current_sentence.empty()) current_sentence += " ";
                            current_sentence += winner;
                            last_prediction = winner;
                        }

                        std::cout << ">>> ADDED TO BATCH: [ " << winner << " ] (Candidates: ";
                        for(auto& opt : wc.options) std::cout << opt << " ";
                        std::cout << ")" << std::endl;
                    }
                    current_sign_buffer.clear();
                } 
                else if (data.contains("type") && data["type"] == "FINISH_BATCH") {
                    std::cout << "\n[BRIDGE] Saving lattice and triggering AI..." << std::endl;
                    
                    std::ofstream outFile("bridge_lattice.txt");
                    if (outFile.is_open()) {
                        for (size_t i = 0; i < sentence_lattice.size(); ++i) {
                            outFile << "Word " << (i+1) << ": { ";
                            for (size_t j = 0; j < sentence_lattice[i].options.size(); ++j) {
                                outFile << sentence_lattice[i].options[j] << (j == sentence_lattice[i].options.size()-1 ? "" : ", ");
                            }
                            outFile << " }" << std::endl;
                        }
                        outFile.close();
                        
                        // TRIGGER THE PYTHON BRIDGE
                        std::cout << "[BRIDGE] Calling Python AI Polisher..." << std::endl;
                        system("python ai_polisher.py --auto");
                    } else {
                        std::cerr << "[ERROR] Could not write to bridge_lattice.txt" << std::endl;
                    }
                    
                    sentence_lattice.clear();
                    current_sentence = "";
                    last_prediction = "";
                }
                else if (data.contains("type") && data["type"] == "FRAME") {
                    // --- 5. PARSE LIVE DATA ---
                    Frame frame;
                    frame.timestamp = data.value("timestamp", 0.0);
                    
                    // Hands
                    if (data.contains("hands") && data["hands"].is_array()) {
                        HandData left_hd, right_hd;
                        left_hd.is_present = false;
                        right_hd.is_present = false;
                        
                        for (const auto& h : data["hands"]) {
                            HandData hd;
                            hd.wrist_pos = {h["wrist_pos"]["x"], h["wrist_pos"]["y"], h["wrist_pos"]["z"]};
                            if (h.contains("landmarks") && h["landmarks"].is_array() && h["landmarks"].size() > 0) {
                                hd.is_present = true;
                                for (const auto& lm : h["landmarks"]) {
                                    hd.landmarks.push_back({lm["x"], lm["y"], lm["z"]});
                                }
                            } else {
                                hd.is_present = false;
                            }
                            
                            if (h.contains("label") && h["label"] == "Right") {
                                right_hd = hd;
                            } else {
                                left_hd = hd; // Default to Left
                            }
                        }
                        frame.hands.push_back(left_hd);
                        frame.hands.push_back(right_hd);
                    }

                    // Face (8 points)
                    if (data.contains("face") && !data["face"].is_null()) {
                        auto f = data["face"];
                        frame.face.forehead = {f["forehead"]["x"], f["forehead"]["y"], f["forehead"]["z"]};
                        frame.face.chin = {f["chin"]["x"], f["chin"]["y"], f["chin"]["z"]};
                        frame.face.nose = {f["nose"]["x"], f["nose"]["y"], f["nose"]["z"]};
                        frame.face.l_cheek = {f["l_cheek"]["x"], f["l_cheek"]["y"], f["l_cheek"]["z"]};
                        frame.face.r_cheek = {f["r_cheek"]["x"], f["r_cheek"]["y"], f["r_cheek"]["z"]};
                        frame.face.mouth = {f["mouth"]["x"], f["mouth"]["y"], f["mouth"]["z"]};
                        frame.face.l_eye = {f["l_eye"]["x"], f["l_eye"]["y"], f["l_eye"]["z"]};
                        frame.face.r_eye = {f["r_eye"]["x"], f["r_eye"]["y"], f["r_eye"]["z"]};
                        frame.has_face = true;
                    }

                    // Pose (Ears & Shoulders)
                    if (data.contains("pose_anchors") && !data["pose_anchors"].is_null()) {
                        auto p = data["pose_anchors"];
                        frame.pose.l_ear = {p["l_ear"]["x"], p["l_ear"]["y"], p["l_ear"]["z"]};
                        frame.pose.r_ear = {p["r_ear"]["x"], p["r_ear"]["y"], p["r_ear"]["z"]};
                        frame.pose.l_shoulder = {p["l_shoulder"]["x"], p["l_shoulder"]["y"], p["l_shoulder"]["z"]};
                        frame.pose.r_shoulder = {p["r_shoulder"]["x"], p["r_shoulder"]["y"], p["r_shoulder"]["z"]};
                        frame.has_pose = true;
                    }

                    current_sign_buffer.push_back(frame);
                }
            } catch (const std::exception& e) {
                std::cerr << "\n[JSON ERROR] " << e.what() << std::endl;
            }
        }

        // --- KEYBOARD HANDLING (Non-Blocking) ---
        if (GetAsyncKeyState('F') & 0x8000) {
            std::cout << "\n\n======================================================\n";
            std::cout << "         --- BATCH FINISHED: FINAL LATTICE ---         \n";
            std::cout << "======================================================\n";
            
            for (size_t i = 0; i < sentence_lattice.size(); ++i) {
                std::cout << "Word " << (i+1) << ": { ";
                for (size_t j = 0; j < sentence_lattice[i].options.size(); ++j) {
                    std::cout << sentence_lattice[i].options[j] << (j == sentence_lattice[i].options.size()-1 ? "" : ", ");
                }
                std::cout << " }" << std::endl;
            }
            
            std::cout << "\n[AI PROMPT READY]: Copy the list above to Gemini/ChatGPT \n";
            std::cout << "to reconstruct the perfect sentence.\n";
            std::cout << "======================================================\n\n";
            
            sentence_lattice.clear();
            current_sentence = "";
            last_prediction = "";
            Sleep(1000); // Debounce
        }
    }
    return 0;
}
