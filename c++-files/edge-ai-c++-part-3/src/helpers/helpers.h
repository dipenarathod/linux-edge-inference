#pragma once
#include <string>
#include <vector>
#include <utility>

namespace helpers{
    // Load model labels from a text file, one label per line
    std::vector<std::string> load_labels(const std::string& labels_file_path);

    // Comparison function to sort pairs of (index, score) in descending order by score
    bool compare_by_scores_descending(const std::pair<size_t, float>& lhs,
                                const std::pair<size_t, float>& rhs);

    // Structure to hold process memory usage metrics
    struct process_memory {
        size_t virtual_mem_mb; // Total virtual memory size (VSIZE) allocated to the process.
        size_t resident_mem_mb; // The actual physical memory (RAM) the process is currently using.
        size_t shared_mem_mb; // The number of resident pages that are shared with other processes (such as shared libraries like libc).
    };

    // Reads /proc/self/statm and returns RAM metrics in Megabytes (MB)
    // All values in this file are displayed as a single line of space-separated numbers, 
    // and they are measured in page units rather than bytes. 
    // On a standard Raspberry Pi Linux OS, one page is typically 4 KB (4096 bytes).
    // _SC_PAGESIZE is defined as the size of a single page on the system.
    process_memory get_current_memory_usage();

    // Structure to hold time metrics for inference
    struct time_metrics {
        double mean_latency_ms; // Mean latency in milliseconds
        double p50_latency_ms;  // P50 latency in milliseconds
        double p90_latency_ms;  // P90 latency in milliseconds
        double p95_latency_ms;  // P95 latency in milliseconds
        double p99_latency_ms;  // P99 latency in milliseconds
    };

    // Function to calculate mean and P99 latency from a vector of run times
    time_metrics calculate_latency_metrics(std::vector<double>& run_times_ms);

    // Function to calculate average memory usage from a vector of process_memory structures
    process_memory calculate_average_memory_usage(const std::vector<process_memory>& memory_usages);
};