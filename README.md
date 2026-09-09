Edge AI inference project

## Python Files
- `export_mobilenetv4.py`: This script loads a MobileNetV3 model from Hugging Face, converts it to ONNX format, and validates the graph integrity.

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

### You can use the VS Code GUI to make folders and edit files. I am showing the commands below to learn them better.

7. Create a CMake file in the project folder
'''admin@raspberrypi5:~/edge-ai-c++ $ nano CMakeLists.txt'''

You can find the contents of the CMake file below here: TODO

8. Create a src directory in the project folder
'''admin@raspberrypi5:~/edge-ai-c++ $ mkdir src
admin@raspberrypi5:~/edge-ai-c++ $ cd src'''


9. Create main.cpp in the src sub-directory
'''admin@raspberrypi5:~/edge-ai-c++/src $ nano main.cpp'''

For now, just make a single line comment in the main file, like: '''//TODO'''

10. Create a models folder in the project folder
'''
admin@raspberrypi5:~/edge-ai-c++ $ mkdir models
admin@raspberrypi5:~/edge-ai-c++ $ ls
CMakeLists.txt  models  src
'''

11. Copy the generated ONNX model to the models folder using the VS Code GUI.

