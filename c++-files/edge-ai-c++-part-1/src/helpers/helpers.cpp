#include <iostream>
#include <fstream>
#include <vector>
#include "helpers.h"

std::vector<std::string> helpers::load_labels(const std::string& labels_file_path) {
    std::vector<std::string> labels;
    std::ifstream file(labels_file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open labels file: " + labels_file_path);
    }
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
}

// Comparator function
bool helpers::compare_by_scores_descending(const std::pair<size_t, float>& lhs,
                               const std::pair<size_t, float>& rhs) {
    return lhs.second > rhs.second;
}