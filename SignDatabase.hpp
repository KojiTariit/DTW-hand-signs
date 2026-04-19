#pragma once
#include <iostream>
#include <filesystem>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include "json.hpp"
#include "DtwEngine.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

struct SignTemplate {
    std::string name;
    std::string category;
    std::string hand_count;
    std::vector<std::vector<float>> features;
};

class SignDatabase {
public:
    std::vector<SignTemplate> templates;

    void loadFromDirectory(const std::string& rootPath) {
        templates.clear();
        std::cout << "--- Scanning Database: " << rootPath << " ---" << std::endl;

        if (!fs::exists(rootPath)) {
            std::cerr << "!! Error: Directory does not exist: " << rootPath << std::endl;
            return;
        }

        for (const auto& entry : fs::recursive_directory_iterator(rootPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                
                std::string signName = entry.path().stem().string();
                std::string hand_count = entry.path().parent_path().filename().string();
                std::string category = entry.path().parent_path().parent_path().filename().string();
                
                std::vector<Frame> sequence = loadJsonFile(entry.path().string());
                
                if (!sequence.empty()) {
                    auto features = DtwEngine::extractFeatures(sequence);
                    templates.push_back({signName, category, hand_count, features});
                    std::cout << "[LOADED] " << category << "/" << hand_count << "/" << signName << " (" << sequence.size() << " frames)" << std::endl;
                }
            }
        }
        
        std::cout << "--- Scan Complete. Total Signs: " << templates.size() << " ---" << std::endl;
    }

private:
    std::vector<Frame> loadJsonFile(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return {};
        
        try {
            json data = json::parse(f);
            std::vector<Frame> seq;
            for (const auto& item : data) {
                Frame frame;
                frame.timestamp = item.value("timestamp", 0.0);
                
                // NEW FORMAT: "hands" is an array
                if (item.contains("hands") && item["hands"].is_array()) {
                    for (const auto& h : item["hands"]) {
                        HandData hd;
                        hd.wrist_pos = {h["wrist_pos"]["x"], h["wrist_pos"]["y"], h["wrist_pos"]["z"]};
                        for (const auto& lm : h["landmarks"]) {
                            hd.landmarks.push_back({lm["x"], lm["y"], lm["z"]});
                        }
                        frame.hands.push_back(hd);
                    }
                } 
                // OLD FORMAT FALLBACK: "landmarks" is at top level
                else if (item.contains("landmarks")) {
                    HandData hd;
                    hd.wrist_pos = {item["wrist_pos"]["x"], item["wrist_pos"]["y"], item["wrist_pos"]["z"]};
                    for (const auto& lm : item["landmarks"]) {
                        hd.landmarks.push_back({lm["x"], lm["y"], lm["z"]});
                    }
                    frame.hands.push_back(hd);
                }
                seq.push_back(frame);
            }
            return seq;
        } catch (...) {
            return {};
        }
    }
};
