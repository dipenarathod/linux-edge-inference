#include <iostream>
#include "preprocess.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
namespace preprocess{
    // List image files in a directory
    std::vector<std::string> list_images(const std::string& dir){
        std::filesystem::path root_dir = dir;
        std::vector<std::string> images;
        // Traverse recursively through all subdirectories
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root_dir)){
            if (entry.is_regular_file()){
                images.push_back(entry.path().string());
            }
        }
        std::cout<<"Found "<<images.size()<<" images in directory: "<<dir<<std::endl;
        return images;
    }

    // Preprocess a single image into a flattened NCHW float32 vector, 
    // ready to hand straight to Ort::Value::CreateTensor
    std::vector<float> preprocess_image(const std::string& image_path){
        // To get the image in RGB, we must first load the image in BGR and then convert it to RGB.
        // 1. Load the image in BGR
        // cv::Mat is the primary data structure used to 
        // store and manipulate multi-dimensional numerical arrays
        cv::Mat img_bgr = cv::imread(image_path, cv::IMREAD_COLOR);
        if (img_bgr.empty()) {
            // std::cerr<<"Error: Could not read image file: "<<image_path<<std::endl;
            throw std::runtime_error("Could not read image file: " + image_path); // To stop execution
        }

        // 2. Convert BGR to RGB (as required by most TensorFlow/ONNX models)
        cv::Mat img_rgb;
        cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

        // 3. Resize image based on crop percentage
        // The above line is redundant as our images are square, but let's do it for portability
        // Target size set as 224 in the header file
        int resize_size = static_cast<int>(std::round(target_size / crop_pct));
        int width = img_rgb.cols;
        int height = img_rgb.rows;
        int new_w, new_h;

        if (width < height) {
            new_w = resize_size;
            new_h = static_cast<int>(std::round(height * (double)resize_size / width));
        } else {
            new_h = resize_size;
            new_w = static_cast<int>(std::round(width * (double)resize_size / height));
        }

        // IMPORTANT
        cv::Mat img_resized;
        // Use INTER_CUBIC to match 'bicubic' interpolation
        cv::resize(img_rgb, img_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

        // 4. Center Crop to 224x224
        int crop_x = (img_resized.cols - target_size) / 2;
        int crop_y = (img_resized.rows - target_size) / 2;
        cv::Rect crop_region(crop_x, crop_y, target_size, target_size);
        cv::Mat img_cropped = img_resized(crop_region);

        // 5. Convert to Float and scale to [0.0, 1.0]
        cv::Mat img_float;
        img_cropped.convertTo(img_float, CV_32FC3, 1.0 / 255.0);

        // 6. Normalize with Mean and Std Dev
        // Core math: (pixel - mean) / std
        cv::subtract(img_float, mean, img_float);
        cv::divide(img_float, std, img_float);

        // 7. HWC to CHW Layout Conversion
        // ONNX Runtime expects [Channels, Height, Width] instead of OpenCV's [Height, Width, Channels]
        // HWC -> CHW, flatten into a single NCHW (batch=1) float vector.
        std::vector<float> input_tensor_values(1 * 3 * target_size * target_size);
        std::vector<cv::Mat> chw_channels(3);
        
        // Set up pointers inside the output vector for each channel
        for (int i = 0; i < 3; ++i) {
            chw_channels[i] = cv::Mat(target_size, target_size, CV_32FC1, &(input_tensor_values[i * target_size * target_size]));
        }
        
        // Split the interleaved HWC image into planar CHW maps
        cv::split(img_float, chw_channels);

        return input_tensor_values;

    }
}