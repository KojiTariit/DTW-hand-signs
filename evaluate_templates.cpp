#include "SignDatabase.hpp"
#include "DtwEngine.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>

void evaluateCategory(SignDatabase& db, const std::string& category) {
    auto& templates = db.categorized_templates[category];
    if (templates.empty()) return;

    std::cout << "\n==================================================\n";
    std::cout << " EVALUATING CATEGORY: " << category << "\n";
    std::cout << "==================================================\n";
    std::cout << std::left << std::setw(20) << "Sign Name" 
              << std::setw(15) << "Nearest Conf." 
              << std::setw(12) << "Score" 
              << "Status\n";
    std::cout << "--------------------------------------------------\n";

    for (auto const& [name, features] : templates) {
        float best_competitor_score = 999999.0f;
        std::string nearest_competitor = "NONE";

        for (auto const& [other_name, other_features] : templates) {
            if (name == other_name) continue;

            float score = DtwEngine::computeDualScore(features, other_features, 0.4f);
            if (score < best_competitor_score) {
                best_competitor_score = score;
                nearest_competitor = other_name;
            }
        }

        std::string status = "OK";
        if (best_competitor_score < 0.20f) status = "CRITICAL (TOO CLOSE)";
        else if (best_competitor_score < 0.35f) status = "WARNING (CLOSE)";

        std::cout << std::left << std::setw(20) << name 
                  << std::setw(15) << nearest_competitor 
                  << std::fixed << std::setprecision(3) << std::setw(12) << best_competitor_score 
                  << status << "\n";
    }
}

int main() {
    SignDatabase db;
    std::cout << "Loading templates...\n";
    db.loadFromDirectory("templates");
    
    evaluateCategory(db, "movement/single_hand");
    evaluateCategory(db, "movement/2_hands");
    evaluateCategory(db, "static/single_hand");

    return 0;
}
