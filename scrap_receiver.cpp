#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <vector>
#include <cmath>
#include "json.hpp"
#include "SignDatabase.hpp"
#include "ml_project/StaticSignClassifier.hpp"

using json = nlohmann::json;

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
// Extracts exactly what m2cgen expects (77 features)
std::vector<float> extract_ml_features(const Frame& f) {
    std::vector<float> features;
    if (f.hands.empty()) return features;

    const auto& hand = f.hands[0];
    const auto& lms = hand.landmarks;
    
    Point3D p0 = lms[0];
    Point3D p9 = lms[9];
    float hand_size = magnitude(sub_points(p9, p0));
    if (hand_size < 1e-6f) hand_size = 1.0f;

    // 1. Wrist-Relative Distances (20 Features)
    for (int i = 1; i < 21; ++i) {
        features.push_back(magnitude(sub_points(lms[i], p0)) / hand_size);
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

    // 6. Thumb-Cross Matrix (8 Features) - Solves A vs S (PIPs) and M/N/T (MCPs)
    int cross_pips[] = {6, 10, 14, 18};
    int cross_mcps[] = {5, 9, 13, 17};
    for (int i = 0; i < 4; ++i) {
        features.push_back(magnitude(sub_points(lms[4], lms[cross_pips[i]])) / hand_size);
    }
    for (int i = 0; i < 4; ++i) {
        features.push_back(magnitude(sub_points(lms[4], lms[cross_mcps[i]])) / hand_size);
    }

    return features;
}

#pragma comment(lib, "ws2_32.lib")

int main() {
    std::cout << "--- ROOT SCRAP RECEIVER ONLINE ---" << std::endl;
    
    // 1. Load the templates (Load ONLY your movement files; Static is already in the brain!)
    SignDatabase db;
    db.loadFromDirectory("c:/Users/USER/Desktop/DTW/templates");

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

    while (true) {
        int bytesReceived = recvfrom(recvSocket, buffer, sizeof(buffer) - 1, 0, (sockaddr*)&senderAddr, &senderAddrSize);
        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0';
            std::string message(buffer);
            
            try {
                json data = json::parse(message);
                
                if (data.contains("type") && data["type"] == "END_OF_SIGN") {
                    std::cout << "\n[SIGN COMPLETE] Processing " << current_sign_buffer.size() << " frames." << std::endl;
                    
                    if (current_sign_buffer.size() >= 5) {
                        // --- 3. ANALYZE MOVEMENT ---
                        size_t max_hands = 0;
                        float max_wrist_dist = 0.0f;
                        float max_shape_variance = 0.0f;
                        
                        std::vector<Point3D> start_wrist; 
                        std::vector<std::vector<Point3D>> start_shape; 
                        bool init_done = false;

                        for (const auto& f : current_sign_buffer) {
                            max_hands = std::max(max_hands, f.hands.size());
                            if (!init_done && !f.hands.empty()) {
                                for (const auto& h : f.hands) {
                                    start_wrist.push_back(h.wrist_pos);
                                    start_shape.push_back(h.landmarks);
                                }
                                init_done = true;
                            }

                            if (init_done) {
                                for (size_t i = 0; i < f.hands.size() && i < start_wrist.size(); ++i) {
                                    float dx = f.hands[i].wrist_pos.x - start_wrist[i].x;
                                    float dy = f.hands[i].wrist_pos.y - start_wrist[i].y;
                                    float dz = f.hands[i].wrist_pos.z - start_wrist[i].z;
                                    max_wrist_dist = std::max(max_wrist_dist, std::sqrt(dx*dx + dy*dy + dz*dz));

                                    for (size_t j = 0; j < f.hands[i].landmarks.size(); ++j) {
                                        float sx = f.hands[i].landmarks[j].x - start_shape[i][j].x;
                                        float sy = f.hands[i].landmarks[j].y - start_shape[i][j].y;
                                        float sz = f.hands[i].landmarks[j].z - start_shape[i][j].z;
                                        max_shape_variance = std::max(max_shape_variance, std::sqrt(sx*sx + sy*sy + sz*sz));
                                    }
                                }
                            }
                        }

                        // --- 4. SMART ROUTING TREE ---
                        std::string winner = "None";
                        float TH_WRIST = 0.15f; 
                        float TH_SHAPE = 0.06f; 
                        bool is_dynamic = (max_wrist_dist > TH_WRIST || max_shape_variance > TH_SHAPE || max_hands >= 2);

                        if (is_dynamic) {
                            std::cout << "  >> Route: DYNAMIC (DTW Processor w/ Spatial Pruning) <<" << std::endl;
                            float min_dist = 9999.0f;
                            
                            // SPATIAL PRUNING: Determine folder based on hand count (e.g., "movement/2_hands")
                            std::string target_cat = (max_hands >= 2) ? "movement/2_hands" : "movement/single_hand";
                            std::cout << "  >> Filter: Searching [" << target_cat << "] folder only." << std::endl;

                            if (db.categorized_templates.count(target_cat)) {
                                auto live_feat = DtwEngine::extractFeatures(current_sign_buffer);
                                for (auto const& [name, template_feat] : db.categorized_templates.at(target_cat)) {
                                    float dist = DtwEngine::computeDualScore(live_feat, template_feat, 0.4f);
                                    if (dist < min_dist) {
                                        min_dist = dist;
                                        winner = name;
                                    }
                                }
                            } else {
                                std::cerr << "!! Warning: Category [" << target_cat << "] not found in database!" << std::endl;
                            }
                        } else {
                            std::cout << "  >> Route: STATIC (ML Processor) <<" << std::endl;
                            const Frame& mid_f = current_sign_buffer[current_sign_buffer.size() / 2];
                            if (!mid_f.hands.empty()) {
                                std::vector<float> ml_features = extract_ml_features(mid_f);
                                winner = StaticSignClassifier::predict(ml_features);
                            }
                        }
                        
                        std::cout << ">>> PREDICTION: [ " << winner << " ] <<<" << std::endl;
                    }
                    current_sign_buffer.clear();
                } 
                else if (data.contains("type") && data["type"] == "FRAME") {
                    // --- 5. PARSE LIVE DATA ---
                    Frame frame;
                    frame.timestamp = data.value("timestamp", 0.0);
                    
                    // Hands
                    if (data.contains("hands") && data["hands"].is_array()) {
                        for (const auto& h : data["hands"]) {
                            HandData hd;
                            hd.wrist_pos = {h["wrist_pos"]["x"], h["wrist_pos"]["y"], h["wrist_pos"]["z"]};
                            for (const auto& lm : h["landmarks"]) {
                                hd.landmarks.push_back({lm["x"], lm["y"], lm["z"]});
                            }
                            frame.hands.push_back(hd);
                        }
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
    }
    return 0;
}
