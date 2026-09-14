import glob
import timm
import torch
import onnx
import urllib
import numpy as np
from PIL import Image
import onnxruntime.quantization as ort_quant
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

# 8. Pre-process and Quantize FP32 ONNX model to INT8 using ONNX Runtime Quantizer
# ONNX Runtime pre-processing performs shape inference, constant folding, and node fusion.
# Running this before quantization optimizes node patterns for edge execution
# and eliminates the ONNX Runtime pre-processing warning.
onnx_prep_file_path = "mobilenetv3_small_100_lamb_prep.onnx"
onnx_int8_file_path = "mobilenetv3_small_100_lamb_int8_dynamic_quantization.onnx"
print("Running ONNX pre-processing (shape inference and graph optimization)...")
ort_quant.shape_inference.quant_pre_process(
    input_model_path=onnx_file_path,
    output_model_path=onnx_prep_file_path
)

print("Start quantizing pre-processed ONNX model to INT8...")
ort_quant.quantize_dynamic(
    model_input=onnx_prep_file_path,
    model_output=onnx_int8_file_path,
    weight_type=ort_quant.QuantType.QUInt8,
    per_channel=True
)
print(f"Quantized INT8 model saved to {onnx_int8_file_path} successfully.")


# 9. Verify INT8 graph integrity
print("Validating quantized INT8 ONNX graph...")
onnx_int8_model = onnx.load(onnx_int8_file_path)
onnx.checker.check_model(onnx_int8_model)
print("INT8 Graph check passed.")


# 10. Output metadata for C++ integration
input_tensor = onnx_int8_model.graph.input[0]
output_tensor = onnx_int8_model.graph.output[0]

print(f"Input Name:  {input_tensor.name}")
print(f"Output Name: {output_tensor.name}")


# Static quantization (aka post-training quantization) is a more aggressive optimization than dynamic quantization.
# Unlike dynamic quantization (which quantizes weights ahead of time and
# activations on-the-fly at inference), static quantization also fixes the
# activation quantization parameters (scale/zero-point) ahead of time.
# This requires "calibration": running a handful of representative images
# through the FP32 graph to record activation ranges.
# Refer: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

# 12. Get calibration images
# We need a small folder of real, representative input images (NOT random
# noise) so the recorded activation ranges reflect real data distributions.
# A convenient source: https://github.com/EliSchwartz/imagenet-sample-images
# This repo has ~1000 images, one per ImageNet class, a few hundred KB each,
# which is a good default calibration set for an ImageNet-trained model like
# this one. 100-500 images is typically enough; more isn't necessary for a
# must-have baseline. Clone/download it and point CALIBRATION_IMAGE_DIR at
# the resulting folder (or a subset of it), e.g.:
# git clone https://github.com/EliSchwartz/imagenet-sample-images.git
CALIBRATION_IMAGE_DIR = "./calibration_images/imagenet-sample-images"  # Calibration images sub-folder.

# 13. Build the preprocessing pipeline the model itself expects
# Reuse timm's own config/transform so calibration images are resized,
# cropped, and normalized exactly like the model was trained/exported with
# (this must match the ONNX graph's expected input, not an arbitrary resize).
data_config = timm.data.resolve_data_config({}, model=model)  # Resolve data configuration based on the model
preprocess = timm.data.create_transform(**data_config)         # Create a preprocessing transform using the resolved configuration

# See comments below for more details on what data_config and preprocess are

# data_config is just a plain dict of the preprocessing parameters timm associates with this specific model checkpoint — things like:
# {
#     'input_size': (3, 224, 224),
#     'interpolation': 'bicubic',   # or 'bilinear', depends on the checkpoint
#     'mean': (0.485, 0.456, 0.406),
#     'std': (0.229, 0.224, 0.225),
#     'crop_pct': 0.875,
#     'crop_mode': 'center',
#     ...
# }

# timm.data.resolve_data_config({}, model=model) figures these out from the model's pretrained_cfg 
# (metadata bundled with the checkpoint on the Hub) rather than you hardcoding them. This matters because different timm models — 
# even different mobilenetv3 variants — were trained with slightly different mean/std or interpolation, and getting it wrong silently degrades 
# accuracy without throwing an error. The empty dict {} is just "no user overrides," i.e. "trust the model's own config."

# create_transform(**data_config) builds an actual torchvision.transforms.Compose pipeline (resize → center-crop → ToTensor() → Normalize(mean, std)) driven by that dict. 
# So preprocess(image) takes a PIL image of any size and returns a normalized [3, 224, 224] float tensor, matching what the model expects at the pixel-value level, 
# not just the shape. That's why it's used for calibration images specifically: quantization calibration only produces valid activation ranges if 
# the images are fed through in the same distribution (scale, normalization) the model will see at real inference time — 
# feeding it raw unnormalized pixels or a naive resize would skew the calibrated scale/zero-point values.


# NOTE about preprocessing during deployment------------------------------------------------------------------------------------------------------
# We'll need to replicate this exact resize/crop/normalize logic (same interpolation mode, same mean/std) manually, since we won't have timm there. 
# Printing data_config once so we have those exact numbers to hardcode into C++ preprocessing.
print(f"Data config:\n{data_config}")


# 14. Define a CalibrationDataReader
# ONNX Runtime pulls calibration batches through this iterator-like class.
class ImageCalibrationDataReader(ort_quant.CalibrationDataReader):
    def __init__(self, image_dir, input_name, transform, max_images=300):
        image_paths = sorted(
            glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.JPEG")
        )[:max_images]
        if not image_paths:
            raise FileNotFoundError(
                f"No calibration images found in '{image_dir}'. "
                "See the comment above CALIBRATION_IMAGE_DIR for where to get some."
            )
        self.input_name = input_name
        self.transform = transform
        self.image_paths = image_paths
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None
        image = Image.open(self.image_paths[self.index]).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)  # [1, 3, H, W]
        self.index += 1
        return {self.input_name: tensor.numpy().astype(np.float32)}

    def rewind(self):
        self.index = 0


calibration_reader = ImageCalibrationDataReader(
    image_dir=CALIBRATION_IMAGE_DIR,
    input_name=input_tensor.name,   # reuse the FP32 graph's input name from step 6
    transform=preprocess,
)

# 15. Quantize the pre-processed FP32 ONNX model to static INT8
# Reuses onnx_prep_file_path from step 8 (shape-inferred / constant-folded),
# since quantize_static also expects a pre-processed model as input.
onnx_static_int8_file_path = "mobilenetv3_small_100_lamb_int8_static_quantization.onnx"
print("Start static-quantizing pre-processed ONNX model to INT8...")
ort_quant.quantize_static(
    model_input=onnx_prep_file_path,
    model_output=onnx_static_int8_file_path,
    calibration_data_reader=calibration_reader,
    weight_type=ort_quant.QuantType.QInt8,
    activation_type=ort_quant.QuantType.QInt8,
    per_channel=True
)
print(f"Statically quantized INT8 model saved to {onnx_static_int8_file_path} successfully.")

# 16. Verify static INT8 graph integrity
print("Validating statically quantized INT8 ONNX graph...")
onnx_static_int8_model = onnx.load(onnx_static_int8_file_path)
onnx.checker.check_model(onnx_static_int8_model)
print("Static INT8 graph check passed.")

# 17. Output metadata for C++ integration
static_input_tensor = onnx_static_int8_model.graph.input[0]
static_output_tensor = onnx_static_int8_model.graph.output[0]
print(f"Input Name:  {static_input_tensor.name}")
print(f"Output Name: {static_output_tensor.name}")


# 18. Extract label names
# This link shows how to extract the label names from the model:
# https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/models/mobilenet-v3.mdx

url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    f.close()