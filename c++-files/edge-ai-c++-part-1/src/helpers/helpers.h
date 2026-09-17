#pragma once
class helpers{
    public:
        static std::vector<std::string> load_labels(const std::string& labels_file_path);

        static bool compare_by_scores_descending(const std::pair<size_t, float>& lhs,
                                const std::pair<size_t, float>& rhs);
};