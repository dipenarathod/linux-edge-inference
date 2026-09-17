#pragma once

#include <opencv2/opencv.hpp>

namespace preprocess{
    // Got the following by printing the data config in export_static_quantized_mobilenetv4.py
    const int image_width = 224;
    const int image_height = 224;
    const float crop_pct = 0.875f;
    // OpenCV defaults to BGR, while we want RGB. The following values are in RGB
    const cv::Scalar mean(0.485,0.456,0.406); // cv::Scalar is a template class for a 4-element vector. (Unused values default to 0)
    const cv::Scalar std(0.229, 0.224, 0.225);   

    
    // NOTE: For resizing using the crop percentage, we use the smaller edge
    // as the target size. This ensures that the aspect ratio is preserved.
    // Target size is 224, so resize the smaller edge to (224 / 0.875) = 256
    // Does not matter to us as our desired images are square.
    const int target_size = 224;
    
    // List image files in a directory
    std::vector<std::string> list_images(const std::string& dir);

    // Preprocess a single image into a flattened NCHW float32 vector, 
    // ready to hand straight to Ort::Value::CreateTensor
    std::vector<float> preprocess_image(const std::string& image_path);

    
}