#pragma once
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include "json.hpp"

using json = nlohmann::json;

struct DecisionTree {
    std::vector<int> children_left;
    std::vector<int> children_right;
    std::vector<int> feature;
    std::vector<double> threshold;
    std::vector<std::map<int, double>> values; // sparse values: node -> {class_idx: prob}

    void predict_proba(const std::vector<float>& input, std::vector<double>& out_accum) {
        int node = 0;
        // In scikit-learn, leaf nodes have children_left = -1
        while (children_left[node] != -1) {
            int feat = feature[node];
            if (feat >= 0 && feat < static_cast<int>(input.size())) {
                if (input[feat] <= threshold[node]) {
                    node = children_left[node];
                } else {
                    node = children_right[node];
                }
            } else {
                break;
            }
        }
        
        // Accumulate sparse probabilities
        const auto& val_map = values[node];
        for (const auto& pair : val_map) {
            int class_idx = pair.first;
            double prob = pair.second;
            if (class_idx >= 0 && class_idx < static_cast<int>(out_accum.size())) {
                out_accum[class_idx] += prob;
            }
        }
    }
};

class ForestClassifier {
private:
    std::vector<std::string> classes;
    int n_features = 0;
    int n_classes = 0;
    std::vector<DecisionTree> trees;
    bool is_loaded = false;

public:
    ForestClassifier() = default;

    bool load(const std::string& filepath) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            std::cerr << "[ERROR] Could not open model file: " << filepath << std::endl;
            return false;
        }
        try {
            json data = json::parse(f);
            classes = data["classes"].get<std::vector<std::string>>();
            n_features = data["n_features"];
            n_classes = data["n_classes"];
            
            trees.clear();
            for (const auto& t_json : data["trees"]) {
                DecisionTree t;
                t.children_left = t_json["children_left"].get<std::vector<int>>();
                t.children_right = t_json["children_right"].get<std::vector<int>>();
                t.feature = t_json["feature"].get<std::vector<int>>();
                t.threshold = t_json["threshold"].get<std::vector<double>>();
                
                // Read sparse values map
                t.values.clear();
                for (const auto& val_json : t_json["values"]) {
                    std::map<int, double> sparse_val;
                    for (auto it = val_json.begin(); it != val_json.end(); ++it) {
                        sparse_val[std::stoi(it.key())] = it.value().get<double>();
                    }
                    t.values.push_back(sparse_val);
                }
                
                trees.push_back(t);
            }
            is_loaded = true;
            std::cout << "[SUCCESS] Loaded Sparse Forest: " << filepath << " w/ " << trees.size() << " trees, " << n_classes << " classes." << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Parsing forest model failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool loaded() const { return is_loaded; }

    std::vector<std::string> get_classes() const {
        return classes;
    }

    std::vector<double> predict_proba(const std::vector<float>& features) const {
        if (!is_loaded || trees.empty()) return std::vector<double>(n_classes, 0.0);
        
        std::vector<double> accum(n_classes, 0.0);
        for (const auto& tree : trees) {
            const_cast<DecisionTree&>(tree).predict_proba(features, accum);
        }
        
        // Average the predictions across all trees
        double denom = static_cast<double>(trees.size());
        for (int i = 0; i < n_classes; ++i) {
            accum[i] /= denom;
        }
        return accum;
    }

    std::string predict(const std::vector<float>& features) const {
        if (!is_loaded || classes.empty()) return "None";
        auto probs = predict_proba(features);
        int max_idx = 0;
        double max_val = probs[0];
        for (int i = 1; i < n_classes; ++i) {
            if (probs[i] > max_val) {
                max_val = probs[i];
                max_idx = i;
            }
        }
        return classes[max_idx];
    }
};
