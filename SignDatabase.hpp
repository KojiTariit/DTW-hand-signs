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

class SignDatabase {
public:
    // A map partitioned by CATEGORY (e.g., "movement", "static")
    // categorize_templates["static"]["A"] = ...
    std::map<std::string, std::map<std::string, std::vector<std::vector<float>>>> categorized_templates;

    void loadFromDirectory(const std::string& rootPath) {
        categorized_templates.clear();
        std::cout << "--- Scanning Database: " << rootPath << " ---" << std::endl;

        if (!fs::exists(rootPath)) {
            std::cerr << "!! Error: Directory does not exist: " << rootPath << std::endl;
            return;
        }

        // Recursive directory iterator finds files in EVERY subfolder!
        for (const auto& entry : fs::recursive_directory_iterator(rootPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                
                std::string signName = entry.path().stem().string();
                std::string category = entry.path().parent_path().filename().string();
                
                std::vector<Frame> sequence = loadJsonFile(entry.path().string());
                
                if (!sequence.empty()) {
                    categorized_templates[category][signName] = DtwEngine::extractFeatures(sequence);
                    std::cout << "[LOADED] " << category << "/" << signName << " (" << sequence.size() << " frames)" << std::endl;
                }
            }
        }
        
        int total_signs = 0;
        for (const auto& cat : categorized_templates) total_signs += cat.second.size();
        std::cout << "--- Scan Complete. Total Signs: " << total_signs << " ---" << std::endl;
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
