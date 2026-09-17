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

int main(){
    const std::string model_path = std::string(MODEL_DIR) + "mobilenetv3_small_100_lamb.onnx";
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
        std::vector<std::pair<size_t, float>> indexValuePairs;
        for (size_t i = 0; i < output_element_count; i++) {
            indexValuePairs.emplace_back(i, output_data[i]);
        }
        
        // 12. Sort the pairs by score in descending order
        std::sort(indexValuePairs.begin(), indexValuePairs.end(), helpers::compare_by_scores_descending);

        // 13. Print Top 5, mapping index -> label
        std::cout << "--- Top 5 Predictions ---" << std::endl;
        for (size_t i = 0; i < 5; i++) {
            const auto& result = indexValuePairs[i];
            std::cout << i + 1 << ": " << labels[result.first]
                    << " (score: " << result.second << ")" << std::endl;
        }

        // Part 1 End ------------------------------------------------------------------------

        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl;
            return -1;
        } catch (const std::exception& e) {
            std::cerr << "Standard Exception: " << e.what() << std::endl;
            return -1;
        }
    return 0;
}
