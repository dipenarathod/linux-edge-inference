import timm
import torch
import onnx
# 1. Model name
# Model link: https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k
# (8th Sept, 2026, press the "Use this model" button > Use in timm to get the model name)
# I got the "hf:hub" prefix from there. 
model_name = 'hf_hub:timm/mobilenetv3_small_100.lamb_in1k'

# 2. Load the model and put it into evaluation mode
print("Start loading the model...")
model = timm.create_model(model_name, pretrained=True)
model.eval()

print(f"Model {model_name} loaded successfully.")

# 3. Create dummy input matching ImageNet standard dimensions (FP32)
# Shape: [Batch Size, Channels, Height, Width] -> [1, 3, 224, 224]
# Refer to: https://github.com/microsoft/olive-recipes/blob/main/timm-mobilenetv3_small_100.lamb_in1k/olive/config.json
# In the document, we have:
# {
#     "input_model": {
#         "type": "PytorchModel",
#         "model_path": "timm/mobilenetv3_small_100.lamb_in1k",
#         "model_loader": "load_timm",
#         "model_script": "user_script.py",
#         "io_config": {
#             "input_names": [ "x" ],
#             "input_shapes": [ [ 1, 3, 224, 224 ] ],
#             "output_names": [ "output" ],
#             "dynamic_axes": { "x": { "0": "batch_size" }, "output": { "0": "batch_size" } }
#         }
#     },
# ...
batch_size = 1 # The number of images processed at the exact same time
dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)

# 4. Define the output file name
onnx_file_path = "mobilenetv3_small_100_lamb.onnx"

print("Start exporting the model to ONNX format...")

# PyTorch ONNX export function: https://docs.pytorch.org/docs/2.14/onnx_export.html#torch.onnx.export
# 5. Export using the standard PyTorch ONNX exporter
dynamic_axes = {
    'input_images': {0: 'batch_size', 2: 'height', 3: 'width'}, # Allow dynamic batch, height, and width
    'output_predictions': {0: 'batch_size'}                     # Output batch size must match input
}
torch.onnx.export(
    model = model,                      # Model being run
    args = dummy_input,                # Model input (or a tuple for multiple inputs)
    f = onnx_file_path,             # Where to save the model
    export_params = True,         # Store the trained parameter weights inside the model file
    opset_version = 14,   
    dynamo = False,
    input_names=['input_images'],    # The model's input names  
    output_names=['output_predictions'], # The model's output names
    dynamic_axes=dynamic_axes,
    
)

# For opset version, I googled "recommended onnx opset version for running quantized mobilenetv3 in C++ on Raspberry Pi 4"
# and the search results suggested at least 13

print(f"Model exported to {onnx_file_path} successfully.")


# 6. Verify graph integrity
print("Validating exported ONNX graph...")
onnx_model = onnx.load(onnx_file_path)
onnx.checker.check_model(onnx_model)
print("Graph check passed.")

# 7. Output metadata for C++ integration
input_tensor = onnx_model.graph.input[0]
output_tensor = onnx_model.graph.output[0]

print(f"Input Name:  {input_tensor.name}")
print(f"Output Name: {output_tensor.name}")

# If dynamo is not set to False, we get the following error
# If dynamo is set to False, we must specify dynamic_axes in the export function.
# The model version conversion is not supported by the onnxscript version converter and fallback is enabled. The model will be converted using the onnx C API (target version: 14).
# Failed to convert the model to the target version 14 using the ONNX C API. The model was not modified
# Traceback (most recent call last):
#   File "/home/dipen/Desktop/linux-edge-inference/python-files/.venv/lib/python3.12/site-packages/onnxscript/version_converter/__init__.py", line 137, in call
#     converted_proto = _c_api_utils.call_onnx_api(
#                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/dipen/Desktop/linux-edge-inference/python-files/.venv/lib/python3.12/site-packages/onnxscript/version_converter/_c_api_utils.py", line 65, in call_onnx_api
#     result = func(proto)
#              ^^^^^^^^^^^
#   File "/home/dipen/Desktop/linux-edge-inference/python-files/.venv/lib/python3.12/site-packages/onnxscript/version_converter/__init__.py", line 132, in _partial_convert_version
#     return onnx.version_converter.convert_version(
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/dipen/Desktop/linux-edge-inference/python-files/.venv/lib/python3.12/site-packages/onnx/version_converter.py", line 39, in convert_version
#     converted_model_str = C.convert_version(model_str, target_version)
#                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# RuntimeError: /project/onnx/version_converter/adapters/axes_input_to_attribute.h:56: adapt: Assertion `node->hasAttribute(kaxes)` failed: No initializer or constant input to node found

