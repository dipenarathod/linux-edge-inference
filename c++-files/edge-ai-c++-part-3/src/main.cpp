// Useful links: 
// https://onnxruntime.ai/docs/tutorials/iot-edge/rasp-pi-cv.html
// https://github.com/sudhirsilwal23/ONNX-Tutorial-CPP/tree/main
// https://www.youtube.com/watch?v=imjqRdsm2Qw
// https://dev.to/wolfram27/setting-up-and-using-onnx-runtime-for-c-in-linux-1ho9
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <vector>  
#include <string>
#include <fstream> // For reading labels file
#include "helpers.h"
#include <chrono> // For timing inference
#include "preprocess.h"
#include <filesystem>
int main(){
    const std::string model_path = std::string(MODEL_DIR) + "mobilenetv3_small_100_lamb_int8_static_quantization.onnx";
    // const std::string model_path = std::string(MODEL_DIR) + "mobilenetv3_small_100_lamb_int8_dynamic_quantization.onnx";
    // const std::string model_path = std::string(MODEL_DIR) + "mobilenetv3_small_100_lamb.onnx";
    const std::string model_labels_path = std::string(MODEL_DIR) + "imagenet_classes.txt";
    try{
        std::cout << "Model path: " << model_path << std::endl;
        // 1. Initialize ONNX Runtime Environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MobileNetV3_Loader");

        // 2. Configure Session Options
        Ort::SessionOptions session_options;
        // Lock intra-op thread count to 4 (matching Raspberry Pi 5's 4 physical CPU cores)
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::cout << "Loading model from: " << model_path << std::endl;

        // 3. Create Session (Loads model into memory and compiles graph execution plan)
        Ort::Session session(env, model_path.c_str(), session_options);
        std::cout << "Model loaded successfully into ONNX Runtime session!\n" << std::endl;

        helpers::process_memory memory_after_creating_session = helpers::get_current_memory_usage();
        std::cout << "Virtual Memory (VSIZE): " << memory_after_creating_session.virtual_mem_mb << " MB" << std::endl;
        std::cout << "Resident Memory (RSS): " << memory_after_creating_session.resident_mem_mb << " MB" << std::endl;
        std::cout << "Shared Memory: " << memory_after_creating_session.shared_mem_mb << " MB" << std::endl;


        // 4. Reuse metadata from PyTorch export
        const char* input_names[] = {"input_images"};
        const char* output_names[] = {"output_predictions"};
        const std::vector<int64_t> input_shape = {1, 3, 224, 224};
        const size_t input_tensor_size = 1 * 3 * 224 * 224;

        std::cout << "Model loaded successfully into ONNX Runtime (C++)" << std::endl;
        std::cout << "Input node : " << input_names[0] << " [1, 3, 224, 224]" << std::endl;
        std::cout << "Output node: " << output_names[0] << std::endl;

        // 5. Dummy Input Data (for testing)
        // Since we are using the full model, the input tensor should use float32 pixel values.
        std::vector<float> input_tensor_values(input_tensor_size, 1.0f); // Initialize with ones

        // 6. Define Memory Info (Tells ONNX Runtime where memory lives: CPU)
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, 
            OrtMemType::OrtMemTypeDefault
        );

        // 7. Wrap std::vector into an ONNX Runtime Tensor (Ort::Value)
        // This does not copy memory; it wraps the underlying pointer.
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),  // Pointer to raw CPU float array
            input_tensor_size,           // Total number of elements
            input_shape.data(),          // Pointer to shape array
            input_shape.size()           // Number of dimensions (4)
        );   
        
        // Verify the created tensor is valid
        if (!input_tensor.IsTensor()) {
            std::cerr << "Failed to create Ort::Value input tensor!" << std::endl;
            return -1;
        }

        std::cout << "Running inference..." << std::endl;

        // 8. Execute Inference (session.Run)
        // Signature: session.Run(RunOptions, input_names, input_tensors, num_inputs, output_names, num_outputs)
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},  // Default run options
            input_names,               // Array of input node names
            &input_tensor,             // Array of Ort::Value input tensors
            1,                         // Number of inputs
            output_names,              // Array of output node names
            1                          // Number of outputs
        );

        // 9. Process and Read Output Data
        // output_tensors is a std::vector<Ort::Value> containing execution results
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        
        // Get total number of output elements (1000 for ImageNet classes)
        size_t output_element_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        std::cout << "\n--- Inference Complete ---" << std::endl;
        std::cout << "Output tensor element count: " << output_element_count << std::endl;
        
        // Print first 5 raw output logits
        std::cout << "First 5 output logits: ";
        for (size_t i = 0; i < 5; i++) {
            std::cout << output_data[i] << " ";
        }
        std::cout << std::endl;

        // 10. Load Labels from File
        std::vector<std::string> labels;
        try {
            labels = helpers::load_labels(model_labels_path);
            std::cout << "Loaded " << labels.size() << " labels from: " << model_labels_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading labels: " << e.what() << std::endl;
            return -1;
        }

        // 11. Pair each class index with its score
        // So: (index, output_data[index]) for all 1000 classes
        std::vector<std::pair<size_t, float>> index_value_pairs;
        for (size_t i = 0; i < output_element_count; i++) {
            index_value_pairs.emplace_back(i, output_data[i]);
        }
        
        // 12. Sort the pairs by score in descending order
        std::sort(index_value_pairs.begin(), index_value_pairs.end(), helpers::compare_by_scores_descending);

        // 13. Print Top 5, mapping index -> label
        std::cout << "--- Top 5 Predictions ---" << std::endl;
        for (size_t i = 0; i < 5; i++) {
            const auto& result = index_value_pairs[i];
            std::cout << i + 1 << ": " << labels[result.first]
                    << " (score: " << result.second << ")" << std::endl;
        }

        std::cout<<"Part 1 Complete----------------------------------------------------------\n\n"<<std::endl;
        // Part 1 End ------------------------------------------------------------------------
        // Part 2 Begin ------------------------------------------------------------------------
        std::cout<<"Part 2 and 3 Begin-------------------------------------------------------\n\n"<<std::endl;
        // 14. Measure Inference Time
        auto start = std::chrono::high_resolution_clock::now();
        output_tensors = session.Run(
            Ort::RunOptions{nullptr},  // Default run options
            input_names,               // Array of input node names
            &input_tensor,             // Array of Ort::Value input tensors
            1,                         // Number of inputs
            output_names,              // Array of output node names
            1                          // Number of outputs
        );
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_duration = end - start; // Convert to milliseconds
        std::cout << "Inference time: " << inference_duration.count() << " ms" << std::endl;

        // Get the output data again after timing
        output_data = output_tensors[0].GetTensorMutableData<float>();
        // Get total number of output elements (1000 for ImageNet classes)
        output_element_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        // Print first 5 raw output logits
        // std::cout << "First 5 output logits: ";
        // for (size_t i = 0; i < 5; i++) {
        //     std::cout << output_data[i] << " ";
        // }
        // std::cout << std::endl;

        // 15. Get Current Memory Usage
        helpers::process_memory mem_usage = helpers::get_current_memory_usage();
        std::cout << "\n--- Memory Usage Metrics ---" << std::endl;
        std::cout << "Virtual Memory (VSIZE): " << mem_usage.virtual_mem_mb << " MB" << std::endl;
        std::cout << "Resident Memory (RSS): " << mem_usage.resident_mem_mb << " MB" << std::endl;
        std::cout << "Shared Memory: " << mem_usage.shared_mem_mb << " MB" << std::endl;

        // 16. Run inference 100 times and measure FP32 metrics: (Mean latency, P99 latency, RAM).
        // Mean latency is the average time taken for inference across all runs, while P99 latency is the time taken for the slowest 1% of runs.
        std::vector<double> run_times_ms;
        std::vector<helpers::process_memory> memory_usages;
        // Part 3 changes---------------------------------------------------
        std::vector<std::string> image_paths = preprocess::list_images(std::string(IMAGES_DIR));
        std::vector<double> preprocess_times_ms;
        std::chrono::duration<double, std::milli> preprocess_duration;
        int num_runs = image_paths.size();
        // int num_runs = 10;
        int correct_predictions = 0;
        //-----------------------------------------------------------------
        for (int i = 0; i < num_runs; i++) {

            // Part 3 changes---------------------------------------------------
            // Preprocess images
            auto pre_start = std::chrono::high_resolution_clock::now();
            std::vector<float> input_image_tensor_values = preprocess::preprocess_image(image_paths[i]);
            // std::ofstream out("cpp_preprocessed.bin", std::ios::binary);
            // out.write(reinterpret_cast<const char*>(input_image_tensor_values.data()),
            //           input_image_tensor_values.size() * sizeof(float));
            // float mean_of_input_image_tensor_values = 0;
            // for(const auto val:input_image_tensor_values){
            //     mean_of_input_image_tensor_values += val;
            // }
            // mean_of_input_image_tensor_values /= input_image_tensor_values.size();
            
            // std::cout << "mean: " << mean_of_input_image_tensor_values << std::endl;
            // for (int k = 0; k < 10; k++) std::cout << input_image_tensor_values[k] << " ";
            // Create image input tensor
            Ort::Value input_image_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_image_tensor_values.data(),  // Pointer to raw CPU float array
                input_tensor_size,           // Total number of elements
                input_shape.data(),          // Pointer to shape array
                input_shape.size()           // Number of dimensions (4)
            );   
            auto pre_end = std::chrono::high_resolution_clock::now();
            preprocess_duration = pre_end - pre_start;
            preprocess_times_ms.push_back(preprocess_duration.count());
            //-----------------------------------------------------------------

            auto infer_start = std::chrono::high_resolution_clock::now();
            output_tensors = session.Run(
                Ort::RunOptions{nullptr},  // Default run options
                input_names,               // Array of input node names
                &input_image_tensor,             // Array of Ort::Value input tensors
                1,                         // Number of inputs
                output_names,              // Array of output node names
                1                          // Number of outputs
            );
            auto infer_end = std::chrono::high_resolution_clock::now();
            inference_duration = infer_end - infer_start; // Convert to milliseconds
            // helpers::process_memory end_mem_usage = helpers::get_current_memory_usage();
            // helpers::process_memory delta_mem_usage = {
            //     end_mem_usage.virtual_mem_mb - start_mem_usage.virtual_mem_mb,
            //     end_mem_usage.resident_mem_mb - start_mem_usage.resident_mem_mb,
            //     end_mem_usage.shared_mem_mb - start_mem_usage.shared_mem_mb
            // };
            helpers::process_memory delta_mem_usage = helpers::get_current_memory_usage();
            run_times_ms.push_back(inference_duration.count());
            memory_usages.push_back(delta_mem_usage);

            // Part 3 changes---------------------------------------------------
            // Check if prediction is correct
            std::vector<std::pair<size_t, float>> index_prediction_pairs;
            size_t number_of_outputs = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(); // Get total number of output elements (1000 for ImageNet classes)
            float* inference_output_data = output_tensors[0].GetTensorMutableData<float>(); // output_tensors is a std::vector<Ort::Value> containing execution results

            for (size_t j = 0; j < number_of_outputs; j++) {
                index_prediction_pairs.emplace_back(j, inference_output_data[j]);
            }
            
            // Sort the pairs by score in descending order
            std::sort(index_prediction_pairs.begin(), index_prediction_pairs.end(), helpers::compare_by_scores_descending);

            int ground_truth_label = std::stoi(
                std::filesystem::path(image_paths[i]).parent_path().filename().string()
            ); // The folders are not read sequentially, so we need to find the correct label based on the image path
            // std::cout << "Label: " << static_cast<size_t>(ground_truth_label) << std::endl;
            // std::cout << "Index:" << i << std::endl;
            // std::cout << "Prediction: " << index_prediction_pairs[0].first << std::endl;
            // std::cout << "Number of outputs: " << number_of_outputs << std::endl;
            if(index_prediction_pairs[0].first == static_cast<size_t>(ground_truth_label)) {
                correct_predictions++;
                // std::cout << "Correct Prediction" << std::endl;
            }
            //-----------------------------------------------------------------
        }
        // Part 3 changes---------------------------------------------------
        std::cout << "\n--- Preprocessing Metrics over 1000 runs ---" << std::endl;
        helpers::time_metrics preprocess_metrics = helpers::calculate_latency_metrics(preprocess_times_ms);
        std::cout << "Mean Latency: " << preprocess_metrics.mean_latency_ms << " ms" << std::endl;
        std::cout << "P50 Latency: " << preprocess_metrics.p50_latency_ms << " ms" << std::endl;
        std::cout << "P90 Latency: " << preprocess_metrics.p90_latency_ms << " ms" << std::endl;
        std::cout << "P95 Latency: " << preprocess_metrics.p95_latency_ms << " ms" << std::endl;
        std::cout << "P99 Latency: " << preprocess_metrics.p99_latency_ms << " ms" << std::endl;
        //-----------------------------------------------------------------
        std::cout << "\n--- Inference Metrics over 1000 runs ---" << std::endl;
        helpers::time_metrics latency_metrics = helpers::calculate_latency_metrics(run_times_ms);
        std::cout << "Mean Latency: " << latency_metrics.mean_latency_ms << " ms" << std::endl;
        std::cout << "P50 Latency: " << latency_metrics.p50_latency_ms << " ms" << std::endl;
        std::cout << "P90 Latency: " << latency_metrics.p90_latency_ms << " ms" << std::endl;
        std::cout << "P95 Latency: " << latency_metrics.p95_latency_ms << " ms" << std::endl;
        std::cout << "P99 Latency: " << latency_metrics.p99_latency_ms << " ms" << std::endl;

        std::cout << "\n--- Memory Metrics over 1000 runs ---" << std::endl;
        helpers::process_memory avg_mem_usage = helpers::calculate_average_memory_usage(memory_usages);
        std::cout << "Average Virtual Memory (VSIZE): " << avg_mem_usage.virtual_mem_mb << " MB" << std::endl;
        std::cout << "Average Resident Memory (RSS): " << avg_mem_usage.resident_mem_mb << " MB" << std::endl;
        std::cout << "Average Shared Memory: " << avg_mem_usage.shared_mem_mb << " MB" << std::endl;

        std::cout << "\n--- Accuracy ---" << std::endl;
        std::cout << "Accuracy: " << (static_cast<double>(correct_predictions)*100.0 / num_runs) << "%" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl;
            return -1;
        } catch (const std::exception& e) {
            std::cerr << "Standard Exception: " << e.what() << std::endl;
            return -1;
        }
    return 0;
}
