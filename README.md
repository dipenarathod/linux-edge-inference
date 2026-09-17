Edge AI inference project

## Table of Contents
- [Python Files](#python-files-folder-python-files)
- [Raspberry Pi Setup](#raspberry-pi-setup)
- [Project Roadmap](#project-roadmap)
- [Results](#results)
- [Miscellaneous](#miscellaneous-troubleshooting-tips-etc)

## Python Files Folder (python-files)
- `export_mobilenetv4.py`: This script loads a MobileNetV3 model from Hugging Face, converts it to ONNX format, and validates the graph integrity. 
It also retrieves the labels and stores them locally in "imagenet_classes.txt".\
- `export_dynamic_quantized_mobilenetv4.py`: This script loads a MobileNetV3 model from Hugging Face, **quantizes the model to use INT8 for dynamic quantization**, converts it to ONNX format, and validates the graph integrity. 
It also retrieves the labels and stores them locally in "imagenet_classes.txt".\
- `export_static_quantized_mobilenetv4.py`: This script loads a MobileNetV3 model from Hugging Face, **quantizes the model to use INT8 for static quantization**, converts it to ONNX format, and validates the graph.\
- `shrink_validation_set.py`: This script opens folder **"./imagenetv2-top-images-format-val"** and retain only one image for each class, shrinking the test set size (making it easier to copy the folder to the Raspberry Pi for testing). Refer to **Get a labeled test dataset for validation** section of the README to get more details on how to get the test dataset.\

### Getting the calibration images for static quantization
1. Create a calibration_images folder in the same directory as your Python files.
2. Run the following git clone command:
```bash
(.venv) dipen@dipen-ubuntu-mate:~/Desktop/linux-edge-inference/python-files/calibration_images$ git clone https://github.com/EliSchwartz/imagenet-sample-images.git
```
3. You should have an image folder with 1000+ images in it.

### Get a labeled test dataset for validation
[Image Folder Link](https://huggingface.co/datasets/vaishaal/ImageNetV2/blob/main/imagenetv2-top-images.tar.gz)
1. The zip folder has 1000 folders with each folder name corresponding to a class in the dataset (0, 1, 2 ... 999).\
2. Download the folder and paste it in python-files folder.\
3. Run the Python script (`shrink_validation_set.py`) to retain only one image from each class folder.\
4. You can try copy-pasting the images folder to the Raspberry Pi using the VS Code GUI, but I could not.\
An alternate is to use rsync on your main machine and copy the folder over to your Raspberry Pi:\
```bash
rsync -avzP /path/to/local/folder/ user@remote_host:/path/to/destination/
```
Example use:
```bash
dipen@dipen-ubuntu-mate:~/Desktop/linux-edge-inference/python-files$ rsync -avzP /home/dipen/Desktop/linux-edge-inference/python-files/imagenetv2-top-images-format-val admin@192.168.0.115:/home/admin/edge-ai-c++-part-3
```


## Raspberry Pi Setup

I suggest using the Remote Development Extension in VS Code to connect to your pi over SSH.\
Ensure you are using the a 64-bit OS on the Raspberry Pi.

### Installing C++ ONNX Runtime

1. Make a C++ project directory:\
```bash
admin@raspberrypi5:~ $ mkdir edge-ai-c++
```
2. Set desired ONNX Runtime version
```bash
admin@raspberrypi5:~/edge-ai-c++ $ ORT_VERSION=1.18.0
```
3. Download official Linux ARM64 archive\
```bash
admin@raspberrypi5:~/edge-ai-c++ $ wget https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz
```
4. Extract archive\
```bash
admin@raspberrypi5:~/edge-ai-c++ $ tar -xvf onnxruntime-linux-aarch64-${ORT_VERSION}.tgz
```

5. Move to a standard local directory (~/onnxruntime)\
```bash
admin@raspberrypi5:~/edge-ai-c++ $ mv onnxruntime-linux-aarch64-${ORT_VERSION} ~/onnxruntime
```

6. Clean up archive\
```bash
admin@raspberrypi5:~/edge-ai-c++ $ rm onnxruntime-linux-aarch64-${ORT_VERSION}.tgz
```

### Installing OpenCV
```bash
admin@raspberrypi5:~ $ sudo apt update
admin@raspberrypi5:~ $ sudo apt install -y libopencv-dev
admin@raspberrypi5:~ $ pkg-config --modversion opencv4
4.6.0
```

### Creating a Project Folder on the Raspberry Pi

*You can use the VS Code GUI to make folders and edit files — the commands below are shown for learning purposes.*

1. Create a CMake file in the project folder
```bash
admin@raspberrypi5:~/edge-ai-c++ $ nano CMakeLists.txt
```

You can find the contents of the CMake file in the C++ project folders.

2. Create a src directory in the project folder
```bash
admin@raspberrypi5:~/edge-ai-c++ $ mkdir src
admin@raspberrypi5:~/edge-ai-c++ $ cd src
```

3. Create main.cpp in the src sub-directory
```bash
admin@raspberrypi5:~/edge-ai-c++/src $ nano main.cpp
```

For now, just make a single line comment in the main file, like: ```//TODO```

4. Create a models folder in the project folder
```bash
admin@raspberrypi5:~/edge-ai-c++ $ mkdir models
admin@raspberrypi5:~/edge-ai-c++ $ ls
CMakeLists.txt  models  src
```

5. Copy the generated ONNX model to the models folder using the VS Code GUI (Drag-and-drop).

6. Copy the labels text file to the models folder using the VS Code GUI (Drag-and-drop).

7. The project is split into parts. You can get each part's project folder in this repository. Each part builds on the previous part.

## Project Roadmap

### Part 1: Baseline FP32 Inference
Task 1: Select a small vision model and use Python to export it to an FP32 .onnx file.\

Task 2: Boot Pi 5 and lock the CPU governor to maximum performance (cpufreq-set -g performance) for stable testing.\

Task 3: Set up a C++ project folder with a CMakeLists.txt file and link the ONNX Runtime C++ SDK.\

Task 4: Write basic C++ code to initialize the ONNX Runtime environment and load the FP32 model.\

Task 5: Create a dummy tensor of the same value and run inference.\

Task 6: Read Labels file and print the top-5 highest predictions.\

### Part 2: Benchmarking Harness
Task 1: Write C++ timing logic using std::chrono::steady_clock to measure the exact time inference takes.

Task 2: Write a C++ function to read /proc/self/statm to track RAM usage during execution.

Task 3: Feed a dummy tensor (e.g., an array of ones) into the FP32 model 100 times.

Task 4: Record the baseline FP32 metrics (Mean latency, P99 latency, RAM).

### Part 3: Quantization (INT8)
Focus: Shrink the model and measure the difference.

Task 1: Go back to your Python script and apply Post-Training Quantization (PTQ) to export an INT8 version of your model.

Task 2: Transfer the INT8 .onnx file to the Pi and load it into your C++ harness.

Task 3: Run the exact same 100-run benchmark on the INT8 model.

Task 4: Create a table comparing FP32 vs. INT8 (Latency, RAM, File Size).

Task 5: Introduce a pre-process function to read images frm the prepared validation set and apply the pre-processing steps as obtained by printing the data_config in `export_static_quantized_mobilenetv4.py`.

Task 6: Modify the 100-run benchmark to now run 1000 times, and read one image from each folder. Pre-process the image and pass it to the model for inference. 

## Results

Benchmarked on Raspberry Pi 5, 1000 runs per model, one image per ImageNet class from the validation set.

### Accuracy, Latency, and Memory

| Metric | FP32 | Dynamic INT8 | Static INT8 |
|---|---|---|---|
| **Accuracy** | 69.1% | 69.3% | 61.8% |
| **Preprocessing — Mean (ms)** | 6.3904 | 6.36359 | 6.44299 |
| **Preprocessing — P50 (ms)** | 5.83628 | 5.79783 | 5.85369 |
| **Preprocessing — P90 (ms)** | 7.09122 | 7.03774 | 7.15137 |
| **Preprocessing — P95 (ms)** | 7.91275 | 7.77911 | 7.96448 |
| **Preprocessing — P99 (ms)** | 19.4778 | 20.3605 | 20.416 |
| **Inference — Mean (ms)** | 12.4182 | 11.8773 | 7.19699 |
| **Inference — P50 (ms)** | 11.7996 | 11.5339 | 6.81702 |
| **Inference — P90 (ms)** | 12.5247 | 11.8911 | 7.09615 |
| **Inference — P95 (ms)** | 16.1979 | 12.383 | 9.44353 |
| **Inference — P99 (ms)** | 24.6746 | 22.6549 | 14.8688 |
| **VSIZE — Avg (MB)** | 678 | 659 | 665 |
| **RSS — Avg (MB)** | 109 | 103 | 108 |
| **Shared Mem — Avg (MB)** | 47 | 47 | 47 |

### Percentage Change vs. FP32 Baseline

| Metric | Dynamic INT8 | Static INT8 |
|---|---|---|
| **Inference — Mean** | −4.36% | −42.04% |
| **Inference — P50** | −2.25% | −42.23% |
| **Inference — P90** | −5.06% | −43.34% |
| **Inference — P95** | −23.55% | −41.70% |
| **Inference — P99** | −8.19% | −39.74% |
| **RSS (Avg)** | −5.50% | −0.92% |
| **VSIZE (Avg)** | −2.80% | −1.92% |

Static quantization delivers by far the largest inference speedup (~40%+ across all latency percentiles) at a real but bounded accuracy cost (~7 points); dynamic quantization is essentially accuracy-neutral but yields only a modest speed/memory improvement. Neither quantized model meaningfully reduces process-level resident memory (RSS) — that number is dominated by fixed runtime/session overhead rather than model weight size at this model's scale, even though on-disk file size does shrink for both.

## Miscellaneous (Troubleshooting, tips, etc.)

1. If you change the project folder, delete the build directory, otherwise you won't be able to use CMake:
```bash
admin@raspberrypi5:~/edge-ai-c++-part-1 $ rm -rf build
admin@raspberrypi5:~/edge-ai-c++-part-1 $ mkdir build
admin@raspberrypi5:~/edge-ai-c++-part-1 $ cd build/
admin@raspberrypi5:~/edge-ai-c++-part-1/build $ cmake ..
admin@raspberrypi5:~/edge-ai-c++-part-1/build $ make -j4
```