Edge AI inference project

## Python Files Folder (python-files)
- `export_mobilenetv4.py`: This script loads a MobileNetV3 model from Hugging Face, converts it to ONNX format, and validates the graph integrity. 
It also retrieves the labels and stores them locally in "imagenet_classes.txt".
- `export_dynamic_quantized_mobilenetv4.py`: This script loads a MobileNetV3 model from Hugging Face, **quantizes the model to use INT8 for dynamic quantization**, converts it to ONNX format, and validates the graph integrity. 
It also retrieves the labels and stores them locally in "imagenet_classes.txt".
- 'export_static_quantized_mobilenetv4.py`: This script loads a MobileNetV3 model from Hugging Face, **quantizes the model to use INT8 for static quantization**, converts it to ONNX format, and validates the graph.

### Getting the calibration images for static quantization
1. Create a calibration_images folder in the same directory as your Python files.
2. Run the following git clone command:
'''
(.venv) dipen@dipen-ubuntu-mate:~/Desktop/linux-edge-inference/python-files/calibration_images$ git clone https://github.com/EliSchwartz/imagenet-sample-images.git
'''
3. You should have an image folder with 1000+ images in it.

## Installing C++ ONNX Runtime on Raspberry Pi
I suggest using the Remote Development Extension in VS Code to connect to your pi over SSH.\
Ensure you are using the a 64-bit OS on the Raspberry Pi.

Then, follow these steps:
1. Make a C++ project directory:\
'''admin@raspberrypi5:~ $ mkdir edge-ai-c++'''
2. Set desired ONNX Runtime version
'''admin@raspberrypi5:~/edge-ai-c++ $ ORT_VERSION=1.18.0'''
3. Download official Linux ARM64 archive
'''admin@raspberrypi5:~/edge-ai-c++ $ wget https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz'''
4. Extract archive
'''admin@raspberrypi5:~/edge-ai-c++ $ tar -xvf onnxruntime-linux-aarch64-${ORT_VERSION}.tgz'''

5. Move to a standard local directory (~/onnxruntime)
'''admin@raspberrypi5:~/edge-ai-c++ $ mv onnxruntime-linux-aarch64-${ORT_VERSION} ~/onnxruntime'''

6. Clean up archive
'''admin@raspberrypi5:~/edge-ai-c++ $ rm onnxruntime-linux-aarch64-${ORT_VERSION}.tgz'''

## Installing OpenCV
1. 
'''
admin@raspberrypi5:~ $ sudo apt update
'''

2. 
''' 
admin@raspberrypi5:~ $ sudo apt install -y libopencv-dev
'''

3. 
'''
admin@raspberrypi5:~ $ pkg-config --modversion opencv4
'''

You will get an output like:
'''
4.6.0
'''

## Creating a project folder on the Raspberry Pi

### You can use the VS Code GUI to make folders and edit files. I am showing the commands below to learn them better.

1. Create a CMake file in the project folder
'''admin@raspberrypi5:~/edge-ai-c++ $ nano CMakeLists.txt'''

You can find the contents of the CMake file below here: TODO

2. Create a src directory in the project folder
'''admin@raspberrypi5:~/edge-ai-c++ $ mkdir src
admin@raspberrypi5:~/edge-ai-c++ $ cd src'''


3. Create main.cpp in the src sub-directory
'''admin@raspberrypi5:~/edge-ai-c++/src $ nano main.cpp'''

For now, just make a single line comment in the main file, like: '''//TODO'''

4. Create a models folder in the project folder
'''
admin@raspberrypi5:~/edge-ai-c++ $ mkdir models
admin@raspberrypi5:~/edge-ai-c++ $ ls
CMakeLists.txt  models  src
'''

5. Copy the generated ONNX model to the models folder using the VS Code GUI (Drag-and-drop).

6. Copy the labels text file to the models folder using the VS Code GUI (Drag-and-drop).

7. The project is split into parts. You can get each part's project folder in this repository. Each part builds on the previous part.

### Part 1:
Task 1: Select a small vision model and use Python to export it to an FP32 .onnx file.\

Task 2: Boot Pi 5 and lock the CPU governor to maximum performance (cpufreq-set -g performance) for stable testing.\

Task 3: Set up a C++ project folder with a CMakeLists.txt file and link the ONNX Runtime C++ SDK.\

Task 4: Write basic C++ code to initialize the ONNX Runtime environment and load the FP32 model.\

Task 5: Create a dummy tensor of the same value and run inference.\

Task 6: Read Labels file and print the top-5 highest predictions.\

### Part 2:
Task 1: Write C++ timing logic using std::chrono::steady_clock to measure the exact time inference takes.

Task 2: Write a C++ function to read /proc/self/statm to track RAM usage during execution.

Task 3: Feed a dummy tensor (e.g., an array of ones) into the FP32 model 100 times.

Task 4: Record the baseline FP32 metrics (Mean latency, P99 latency, RAM).

### Part 3:
Focus: Shrink the model and measure the difference.

Task 1: Go back to your Python script and apply Post-Training Quantization (PTQ) to export an INT8 version of your model.

Task 2: Transfer the INT8 .onnx file to the Pi and load it into your C++ harness.

Task 3: Run the exact same 100-run benchmark on the INT8 model.

Task 4: Create a table comparing FP32 vs. INT8 (Latency, RAM, File Size).



## Miscellaneous (Troubleshooting, tips, etc.)

1. If you change the project folder, delete the build directory, otherwise you won't be able to use CMake:
'''
admin@raspberrypi5:~/edge-ai-c++-part-1 $ rm -rf build
admin@raspberrypi5:~/edge-ai-c++-part-1 $ mkdir build
admin@raspberrypi5:~/edge-ai-c++-part-1 $ cd build/
admin@raspberrypi5:~/edge-ai-c++-part-1/build $ cmake ..
admin@raspberrypi5:~/edge-ai-c++-part-1/build $ make -j4
'''


