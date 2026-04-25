#include "SignDatabase.hpp"
#include "DtwEngine.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

int main() {
    SignDatabase db;
    db.loadFromDirectory("templates");
    
    std::vector<std::string> signs = {"dad", "mom", "grandpa", "grandma", "J", "Z"};
    
    std::cout << "\n--- Confusion Matrix (Single Hand) ---\n";
    std::cout << std::setw(10) << "";
    for (const auto& s : signs) std::cout << std::setw(10) << s;
    std::cout << "\n";
    
    for (const auto& row_sign : signs) {
        std::cout << std::setw(10) << row_sign;
        if (db.categorized_templates["movement/single_hand"].count(row_sign) == 0) {
            std::cout << " NOT FOUND\n";
            continue;
        }
        auto feat_row = db.categorized_templates["movement/single_hand"][row_sign];
        
        for (const auto& col_sign : signs) {
            if (db.categorized_templates["movement/single_hand"].count(col_sign) == 0) {
                std::cout << std::setw(10) << "N/A";
                continue;
            }
            auto feat_col = db.categorized_templates["movement/single_hand"][col_sign];
            
            float score = DtwEngine::computeDualScore(feat_row, feat_col, 0.4f);
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << score;
        }
        std::cout << "\n";
    }
    
    return 0;
}
