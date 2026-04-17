//g++ -std=c++17 c:\Users\USER\Desktop\DTW\scrap_receiver.cpp -o c:\Users\USER\Desktop\DTW\scrap_receiver.exe -lws2_32 ; if ($?) { c:\Users\USER\Desktop\DTW\scrap_receiver.exe }


#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <vector>
#include "../SignDatabase.hpp" // Pathed back to root
#include "StaticSignClassifier.hpp" // The generated ML model

float magnitude(const Point3D& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float dot_product(const Point3D& v1, const Point3D& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

Point3D sub_points(const Point3D& a, const Point3D& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

std::vector<float> extract_ml_features(const Frame& f) {
    std::vector<float> features;
    if (f.hands.empty()) return features;

    const auto& hand = f.hands[0];
    const auto& lms = hand.landmarks;
    Point3D p0 = lms[0];
    Point3D p9 = lms[9];
    float hand_size = magnitude(sub_points(p9, p0));
    if (hand_size < 1e-6) hand_size = 1.0f;

    // 1. Normalized Distances
    for (int i = 1; i < 21; ++i) {
        features.push_back(magnitude(sub_points(lms[i], p0)) / hand_size);
    }

    // 2. Joint Angles
    int chains[5][5] = {
        {0, 1, 2, 3, 4}, {0, 5, 6, 7, 8}, {0, 9, 10, 11, 12}, {0, 13, 14, 15, 16}, {0, 17, 18, 19, 20}
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

    // 3. Spread Angles
    int mcp[] = {1, 5, 9, 13, 17};
    for (int i = 0; i < 4; ++i) {
        Point3D v1 = sub_points(lms[mcp[i]], p0);
        Point3D v2 = sub_points(lms[mcp[i+1]], p0);
        float m1 = magnitude(v1);
        float m2 = magnitude(v2);
        features.push_back(dot_product(v1, v2) / (m1 * m2 + 1e-6f));
    }

    return features;
}

#pragma comment(lib, "ws2_32.lib")

int main() {
    // --- 1. DYNAMIC LOADING ---
    // This will now find "A", "I", "J" and ANY other sign you record automatically!
    SignDatabase db;
    db.loadFromDirectory("c:/Users/USER/Desktop/DTW/templates");

    if (db.categorized_templates.empty()) {
        std::cerr << "!! CRITICAL ERROR: No templates found in /templates folder." << std::endl;
        return 1;
    }

    // --- 2. NETWORK SETUP ---
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    SOCKET recvSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    sockaddr_in recvAddr;
    recvAddr.sin_family = AF_INET;
    recvAddr.sin_port = htons(5005);
    recvAddr.sin_addr.s_addr = INADDR_ANY;
    bind(recvSocket, (sockaddr*)&recvAddr, sizeof(recvAddr));

    std::cout << "\n--- DYNAMIC SCRAP RECEIVER ONLINE ---" << std::endl;
    std::cout << "Waiting for sign capture in Python..." << std::endl;

    char buffer[8192];
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
                    if (current_sign_buffer.empty()) continue;

                    if (current_sign_buffer.size() < 5) {
                        std::cout << "!! Error: Sign too short!" << std::endl;
                    } else {
                        // 1. Calculate Statistics
                        size_t max_hands = 0;
                        float max_wrist_dist = 0.0f;
                        float max_shape_variance = 0.0f; // NEW: Detects finger motion even if wrist is still
                        
                        std::vector<Point3D> start_wrist; 
                        std::vector<std::vector<Point3D>> start_shape; // Initial finger positions
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
                                    // A. Wrist Displacement
                                    float dx = f.hands[i].wrist_pos.x - start_wrist[i].x;
                                    float dy = f.hands[i].wrist_pos.y - start_wrist[i].y;
                                    float dz = f.hands[i].wrist_pos.z - start_wrist[i].z;
                                    max_wrist_dist = std::max(max_wrist_dist, std::sqrt(dx*dx + dy*dy + dz*dz));

                                    // B. Shape Variance (Finger movement relative to internal hand frame)
                                    for (size_t j = 0; j < f.hands[i].landmarks.size(); ++j) {
                                        float sx = f.hands[i].landmarks[j].x - start_shape[i][j].x;
                                        float sy = f.hands[i].landmarks[j].y - start_shape[i][j].y;
                                        float sz = f.hands[i].landmarks[j].z - start_shape[i][j].z;
                                        max_shape_variance = std::max(max_shape_variance, std::sqrt(sx*sx + sy*sy + sz*sz));
                                    }
                                }
                            }
                        }

                        std::cout << "\n[DEBUG] Processing Sign (" << current_sign_buffer.size() << " frames)" << std::endl;
                        std::cout << "  - Max Hands Seen: " << max_hands << std::endl;
                        std::cout << "  - Max Wrist Movement: " << max_wrist_dist << std::endl;
                        std::cout << "  - Max Shape Variance: " << max_shape_variance << std::endl;

                        std::string winner = "None";
                        float min_dist = 9999.0f;
                        float TH_WRIST = 0.15f; 
                        float TH_SHAPE = 0.06f; // Sensitivity for finger-only signs like 'J'

                        bool is_dynamic = (max_wrist_dist > TH_WRIST || max_shape_variance > TH_SHAPE || max_hands >= 2);

                        if (is_dynamic) {
                            // --- DYNAMIC PATH (DTW) --- 
                            // Rule: Only search words/movements. Alphabet is for ML only.
                            std::cout << "  >> Routing to DYNAMIC ENGINE (Word) <<" << std::endl;
                            if (db.categorized_templates.count("movement")) {
                                auto live_feat = DtwEngine::extractFeatures(current_sign_buffer);
                                for (auto const& [name, template_feat] : db.categorized_templates.at("movement")) {
                                    float dist = DtwEngine::computeDualScore(live_feat, template_feat, 0.4f); // 0.4 = Prioritize Rhythm (DDTW)
                                    if (dist < min_dist) {
                                        min_dist = dist;
                                        winner = name;
                                    }
                                }
                            }
                        } else {
                            // --- STATIC PATH (ML) ---
                            // Rule: Only use ML for still hand signs.
                            std::cout << "  >> Routing to STATIC MODEL (Alphabet) <<" << std::endl;
                            const Frame& mid_f = current_sign_buffer[current_sign_buffer.size() / 2];
                            if (!mid_f.hands.empty()) {
                                std::vector<float> ml_features = extract_ml_features(mid_f);
                                winner = StaticSignClassifier::predict(ml_features);
                            }
                        }

                        std::cout << ">>> DETECTED: [ " << winner << " ] <<<" << std::endl;
                    }
                    current_sign_buffer.clear();
                } 
                else if (data.contains("type") && data["type"] == "FRAME") {
                    Frame frame;
                    frame.timestamp = data.value("timestamp", 0.0);
                    
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

                    current_sign_buffer.push_back(frame);
                    if (current_sign_buffer.size() % 10 == 0) {
                        std::cout << "." << std::flush;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "\n[JSON ERROR] " << e.what() << std::endl;
            }
        }
    }
    return 0;
}
