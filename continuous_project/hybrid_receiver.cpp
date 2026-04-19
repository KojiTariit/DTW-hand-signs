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
    const auto& lms = f.landmarks;
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

    // 4. Tip Spreads (Normalized tip distances)
    int tips[] = {4, 8, 12, 16, 20};
    for (int i = 0; i < 4; ++i) {
        Point3D t1 = lms[tips[i]];
        Point3D t2 = lms[tips[i+1]];
        features.push_back(magnitude(sub_points(t1, t2)) / hand_size);
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
    
    // Continuous Spelling State Variables
    std::vector<Frame> current_sign_buffer;
    std::vector<std::string> prediction_history;
    std::string last_registered_sign = "";
    int WINDOW_SIZE = 15;
    int HISTORY_SIZE = 10;

    while (true) {
        int bytesReceived = recvfrom(recvSocket, buffer, sizeof(buffer) - 1, 0, (sockaddr*)&senderAddr, &senderAddrSize);
        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0';
            std::string message(buffer);
            
            try {
                json data = json::parse(message);
                
                if (data.contains("type") && data["type"] == "END_OF_SIGN") {
                    std::cout << "[SPACE] " << std::flush;
                    current_sign_buffer.clear();
                    prediction_history.clear();
                    last_registered_sign = "";
                } 
                else if (data.contains("type") && data["type"] == "FRAME") {
                    Frame frame;
                    frame.timestamp = data["timestamp"];
                    frame.wrist_pos = {data["wrist_pos"]["x"], data["wrist_pos"]["y"], data["wrist_pos"]["z"]};
                    for (const auto& lm : data["landmarks"]) {
                        frame.landmarks.push_back({lm["x"], lm["y"], lm["z"]});
                    }
                    
                    // SLIDING WINDOW UPDATE
                    current_sign_buffer.push_back(frame);
                    if (current_sign_buffer.size() > WINDOW_SIZE) {
                        current_sign_buffer.erase(current_sign_buffer.begin()); // Pop oldest frame
                    }

                    // CONTINUOUS EVALUATION
                    if (current_sign_buffer.size() == WINDOW_SIZE) {
                        // 1. Pruning Check (Static vs Move) over last 15 frames
                        float max_wrist_dist = 0;
                        float max_finger_dist = 0;
                        Point3D start_wrist = current_sign_buffer.front().wrist_pos;
                        auto start_lms = current_sign_buffer.front().landmarks;
                        
                        for (const auto& f : current_sign_buffer) {
                            float d_wrist = std::sqrt(std::pow(f.wrist_pos.x - start_wrist.x, 2) + 
                                                      std::pow(f.wrist_pos.y - start_wrist.y, 2));
                            max_wrist_dist = std::max(max_wrist_dist, d_wrist);
                            for (size_t i = 0; i < 21; ++i) {
                                float d_joint = std::sqrt(std::pow(f.landmarks[i].x - start_lms[i].x, 2) + 
                                                          std::pow(f.landmarks[i].y - start_lms[i].y, 2));
                                max_finger_dist = std::max(max_finger_dist, d_joint);
                            }
                        }
                        
                        // If Static -> Route to ML
                        if (max_wrist_dist <= 0.04f && max_finger_dist <= 0.04f) {
                            Frame stable_frame = current_sign_buffer[WINDOW_SIZE / 2];
                            std::vector<float> ml_features = extract_ml_features(stable_frame);
                            std::string winner = StaticSignClassifier::predict(ml_features);
                            
                            // Debouncer logic
                            prediction_history.push_back(winner);
                            if (prediction_history.size() > HISTORY_SIZE) {
                                prediction_history.erase(prediction_history.begin());
                            }
                            
                            if (prediction_history.size() == HISTORY_SIZE) {
                                bool unanimous = true;
                                for (const auto& pred : prediction_history) {
                                    if (pred != winner) unanimous = false;
                                }
                                
                                if (unanimous && winner != last_registered_sign) {
                                    std::cout << winner << std::flush;
                                    last_registered_sign = winner;
                                }
                            }
                        } else {
                            // Movement detected, noise buffer clearing
                            prediction_history.clear();
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "\n[JSON ERROR] " << e.what() << std::endl;
            }
        }
    }
    return 0;
}
