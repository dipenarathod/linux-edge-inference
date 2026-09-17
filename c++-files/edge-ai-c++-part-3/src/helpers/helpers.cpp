#include "helpers.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <unistd.h> // For sysconf to get page size
#include <algorithm> // Required for std::sort
#include <cmath>     // Required for std::ceil
namespace helpers {

    // Load model labels from a text file, one label per line
    std::vector<std::string> load_labels(const std::string& labels_file_path) {
        std::vector<std::string> labels;
        std::ifstream file(labels_file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open labels file: " + labels_file_path);
        }
        std::string line;
        while (std::getline(file, line)) {
            labels.push_back(line);
        }
        file.close();
        return labels;
    }

    // Comparison function to sort pairs of (index, score) in descending order by score
    bool compare_by_scores_descending(const std::pair<size_t, float>& lhs,
                                      const std::pair<size_t, float>& rhs) {
        return lhs.second > rhs.second;
    }

    // Reads /proc/self/statm and returns RAM metrics in Megabytes (MB)
    process_memory get_current_memory_usage() {
        process_memory mem_usage = {0, 0, 0};   
        std::ifstream statm_file("/proc/self/statm");
        if (!statm_file.is_open()) {
            std::cerr << "Error: Could not open /proc/self/statm to read memory usage." << std::endl;
            return mem_usage;
        }
        // Read memory usage from /proc/self/statm
        size_t virtual_mem_pages, resident_mem_pages, shared_mem_pages;
        if(statm_file >> virtual_mem_pages >> resident_mem_pages >> shared_mem_pages) {
            // Convert pages to MB (assuming 4 KB per page)
            size_t page_size_kb = sysconf(_SC_PAGESIZE) / 1024; // Get page size in KB
            mem_usage.virtual_mem_mb = (virtual_mem_pages * page_size_kb) / 1024; // Convert KB to MB
            mem_usage.resident_mem_mb = (resident_mem_pages * page_size_kb) / 1024; // Convert KB to MB
            mem_usage.shared_mem_mb = (shared_mem_pages * page_size_kb) / 1024; // Convert KB to MB
        } else {
            std::cerr << "Error: Could not read memory usage from /proc/self/statm." << std::endl;
        }
        statm_file.close();

        return mem_usage;
    }

    // Function to calculate mean and P99 latency from a vector of run times
    time_metrics calculate_latency_metrics(std::vector<double>& run_times_ms){
        time_metrics metrics = {0.0, 0.0};
        size_t latency_vector_size = run_times_ms.size();
        if (latency_vector_size==0) {
            return metrics;
        }

        // Calculate mean latency
        double sum = 0.0;
        for (size_t i = 0; i < latency_vector_size; i++) {
            sum += run_times_ms[i];
        }
        metrics.mean_latency_ms = sum / latency_vector_size;

        // Sort the run times to find the P99 latency
        std::vector<double> sorted_times = run_times_ms; // Copy to sort
        std::sort(sorted_times.begin(), sorted_times.end());

        // Calculate percentile indices using the standard Nearest-Rank method
        // Because we scale by (n - 1), the max possible index is exactly (n - 1)
        size_t p50_index = std::round(0.50 * (latency_vector_size - 1));
        size_t p90_index = std::round(0.90 * (latency_vector_size - 1));
        size_t p95_index = std::round(0.95 * (latency_vector_size - 1));
        size_t p99_index = std::round(0.99 * (latency_vector_size - 1));

        metrics.p50_latency_ms = sorted_times[p50_index];
        metrics.p90_latency_ms = sorted_times[p90_index];
        metrics.p95_latency_ms = sorted_times[p95_index];
        metrics.p99_latency_ms = sorted_times[p99_index];

        return metrics;
    }

    // Function to calculate average memory usage from a vector of process_memory structures
    process_memory calculate_average_memory_usage(const std::vector<process_memory>& memory_usages){
        size_t total_virtual_mem_mb = 0;
        size_t total_resident_mem_mb = 0;
        size_t total_shared_mem_mb = 0;

        for (const auto& mem_usage : memory_usages) {
            total_virtual_mem_mb += mem_usage.virtual_mem_mb;
            total_resident_mem_mb += mem_usage.resident_mem_mb;
            total_shared_mem_mb += mem_usage.shared_mem_mb;
        }

        process_memory avg_mem_usage = {
            total_virtual_mem_mb / memory_usages.size(),
            total_resident_mem_mb / memory_usages.size(),
            total_shared_mem_mb / memory_usages.size()
        };

        return avg_mem_usage;
    }

}