import glob
import timm
import torch
import onnx
import urllib
import numpy as np
from PIL import Image
import onnxruntime.quantization as ort_quant
import onnxruntime as ort
import os
from collections import Counter
# cv2, qdq_loss_debug, and itertools were added while debugging the static
# quantized model's accuracy collapse:
#   - cv2: needed to reimplement preprocessing in OpenCV so calibration data
#     matches the C++ deployment pipeline exactly (see opencv_preprocess below).
#   - qdq_loss_debug: ONNX Runtime's built-in tool for comparing per-tensor
#     activations between the FP32 and INT8 graphs (see "QDQ Debug Test" section).
#   - itertools: used to glob multiple image extensions in one pass (see
#     quantize_and_eval below).
import cv2
from onnxruntime.quantization import qdq_loss_debug
import itertools


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
    weight_type=ort_quant.QuantType.QInt8,
    op_types_to_quantize=['MatMul', 'Gemm'],  # Excludes Conv layers from dynamic INT8 conversion
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



# DEBUGGING CONTEXT: this function was added after discovering (via byte-level
# diffing of py_preprocessed.bin against cpp_preprocessed.bin, dumped from the
# same source image) that PIL's resize (used by timm's `preprocess` transform
# below) and OpenCV's cv2.resize do NOT produce numerically identical output,
# even with matching interpolation modes. This is because Pillow applies an
# internal box-filter anti-aliasing prefilter when downsampling significantly,
# which OpenCV's INTER_CUBIC does not replicate.
# Since preprocess.cpp is written in OpenCV, static quantization's calibration
# data was being generated from a different pixel distribution than what the
# deployed C++ model actually sees at inference. FP32 and dynamic quantization
# tolerated this mismatch (continuous math degrades gracefully); static
# quantization's fixed INT8 calibration ranges did not.
# Fix: reimplement preprocessing in OpenCV here, matching preprocess.cpp
# step-for-step (same resize-then-crop math, same INTER_CUBIC interpolation),
# so calibration images are decoded and transformed identically to C++.
# Verified: diffing this function's output against cpp_preprocessed.bin gives
# mean abs diff ~0.0006 (floating-point noise), vs. ~0.026 for the old PIL path.
def opencv_preprocess(image_input, target_size=224, crop_pct=0.875):
    # 1. Convert input (PIL Image or path) into an RGB NumPy array
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input)
    else:
        img_rgb = image_input

    # 2. Aspect-preserving resize (matches timm logic)
    resize_size = round(target_size / crop_pct)
    h, w = img_rgb.shape[:2]
    if w < h:
        new_w, new_h = resize_size, round(h * resize_size / w)
    else:
        new_h, new_w = resize_size, round(w * resize_size / h)

    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 3. Center Crop
    x0 = (new_w - target_size) // 2
    y0 = (new_h - target_size) // 2
    img_cropped = img_resized[y0:y0+target_size, x0:x0+target_size]

    # 4. Normalize (ImageNet stats)
    img_float = img_cropped.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std

    # 5. Format to CHW Tensor so .unsqueeze(0) works in get_next()
    chw_array = np.transpose(img_norm, (2, 0, 1))
    return torch.from_numpy(chw_array)

# NOTE: image_input can be a str path, a PIL.Image, or a raw NumPy array.
# When testing this function against cpp_preprocessed.bin, make sure to pass
# the same input TYPE that calibration_reader actually uses (a PIL.Image, via
# the `elif isinstance(image_input, Image.Image)` branch below) - not a file
# path (the `isinstance(image_input, str)` / cv2.imread branch). These two
# branches decode the same JPEG via different libraries (PIL vs libjpeg
# through OpenCV) and can produce slightly different pixel values, so
# verifying the wrong branch gives a false sense of correctness.

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
    def __init__(self, image_dir, input_name, transform, max_images=1000):
        # NOTE: only matches *.jpg / *.JPEG. The imagenetv2 validation set used
        # later in this script (val_image_dir) has *.jpeg (lowercase, four
        # letters) files, which this glob silently skips. Not fixed here since
        # CALIBRATION_IMAGE_DIR (imagenet-sample-images) uses *.JPEG and this
        # reader is only ever pointed at that folder - but if this reader is
        # ever reused against a differently-extensioned folder, apply the same
        # multi-extension glob used in quantize_and_eval below.
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


# DEBUGGING CONTEXT: transform was switched from `preprocess` (timm/PIL) to
# `opencv_preprocess` after confirming the PIL-vs-OpenCV resize mismatch
# described above opencv_preprocess(). This is the fix that made calibration
# data numerically consistent with what preprocess.cpp feeds the model at
# inference. `preprocess` is left commented out for reference/rollback, not
# because it's still a viable option.
calibration_reader = ImageCalibrationDataReader(
    image_dir=CALIBRATION_IMAGE_DIR,
    input_name=input_tensor.name,   # reuse the FP32 graph's input name from step 6
    # transform=preprocess,
    transform = opencv_preprocess
)

# Depthwise convolutions and non-linear activation layers (like Squeeze-and-Excitation blocks) 
# lose critical precision when quantized to INT8. 
# Add a graph parser before static quantization to collect and exclude these nodes.

# DEBUGGING CONTEXT: this function went through several iterations, guided by
# the QDQ Debug Test's per-tensor SQNR output ("xmodel_err" - see bottom of
# script) and by accuracy measured directly in C++/quantize_and_eval:
#   - Originally excluded ALL depthwise convs (group > 1) blanket-wide, on the
#     general heuristic that depthwise conv is sensitive to INT8. This capped
#     accuracy and, worse, left so little of the graph quantized that the
#     static model was actually SLOWER than dynamic quantization (too many
#     isolated int8 islands, no room for QLinearConv/QLinearAdd fusion).
#   - Removed the blanket depthwise exclusion once calibration itself was
#     fixed (Percentile calibrator + opencv_preprocess, both above). This
#     re-crashed accuracy to 6.2%, but a block-by-block sweep (see the sweep
#     loop further down) isolated the actual cause to a SINGLE layer:
#     blocks.0's depthwise conv (not depthwise convs in general).
#   - 'conv_pwl' (linear pointwise projection convs, no activation after them,
#     feeds directly into each block's residual Add) and the SE block's
#     'se.conv_reduce' / 'se.conv_expand' 1x1 convs are excluded because they
#     were the top offenders in the SQNR ranking even after calibration was
#     fixed - not a general heuristic, but on measured data.
#   - extra_exclude_prefix was added specifically to support the block-by-block
#     sweep (see quantize_and_eval usage below): it force-excludes every node
#     under a given /blocks/blocks.N/ prefix, letting each sweep iteration
#     test "what if this one block stayed FP32?" without editing this function.
def get_nodes_to_exclude(onnx_model_path, extra_exclude_prefix=None):
    model = onnx.load(onnx_model_path)
    excluded_nodes = []

    conv_nodes = [n for n in model.graph.node if n.op_type == 'Conv']
    first_conv_name = conv_nodes[0].name if conv_nodes else None

    for node in model.graph.node:
        # First conv operates directly on raw pixel input; highly sensitive
        # to quantization, cheap to keep in FP32.
        if first_conv_name and node.name == first_conv_name:
            excluded_nodes.append(node.name)
            continue

        # block-level override for the sweep
        if extra_exclude_prefix and node.name.startswith(f'/blocks/{extra_exclude_prefix}/'):
            excluded_nodes.append(node.name)
            continue

        # SE-block gating math (ReduceMean/GlobalAveragePool -> ... ->
        # Sigmoid/HardSigmoid -> Mul) rescales feature maps multiplicatively;
        # small quantization error here distorts every channel it touches,
        # not just adds noise. Kept in FP32 regardless of calibration quality
        # (architecturally sensitive, not just calibration-sensitive).
        if node.op_type in ['Sigmoid', 'HardSigmoid', 'Mul', 'Div', 'GlobalAveragePool']:
            excluded_nodes.append(node.name)
            continue

        if node.op_type == 'Conv':
            # conv_pwl: linear projection conv, no activation, feeds Add -
            # top SQNR offender in every block, even post-calibration-fix.
            # se.conv_reduce/se.conv_expand: the SE block's own 1x1 convs -
            # excluding just the elementwise SE ops (Sigmoid/Mul/etc. above)
            # without also excluding these left them "islanded" (INT8 convs
            # sandwiched between FP32 ops), which was worse, not better.
            se_or_sensitive_keywords = (
                'conv_pwl', 'se.conv_reduce', 'se.conv_expand',
                'se_block', 'rrms', 'project', 'expand'
            )
            if any(k in node.name.lower() for k in se_or_sensitive_keywords):
                excluded_nodes.append(node.name)
                continue

        # Final classifier/head layers - low quantization value (tiny
        # fraction of total compute), disproportionate accuracy risk.
        classifier_keywords = ('classifier', 'head', 'fc', 'predictions')
        if any(k in node.name.lower() for k in classifier_keywords):
            excluded_nodes.append(node.name)

    return list(set(excluded_nodes))


# 15. Quantize the pre-processed FP32 ONNX model to static INT8
# Reuses onnx_prep_file_path from step 8 (shape-inferred / constant-folded),
# since quantize_static also expects a pre-processed model as input.
#
# DEBUGGING CONTEXT: this function was built to let the block-sensitivity
# sweep (below) run entirely in Python - quantize a candidate model, eval it,
# repeat - instead of round-tripping through the C++ harness for every
# candidate exclusion set (slow: rebuild + relaunch per config). Only the
# final chosen configuration needs to be validated in C++ against the full
# 1000-image set; this function's accuracy number is a fast proxy for ranking
# candidates against each other, not the final reported figure.
#
# quantize_static options here reflect fixes made during debugging:
#   - calibrate_method=Percentile (not MinMax): MinMax is outlier-sensitive -
#     a single bad calibration pixel can blow out an entire tensor's range.
#     Percentile trims the extremes before fixing scale/zero-point, which is
#     what took the worst xmodel_err from -11.6 dB down to single digits.
#   - activation_type=QUInt8, weight_type=QInt8 (i.e. "U8S8"): the most
#     validated CPU EP scheme for QLinearConv. (Tried QInt8 activations
#     ("S8S8") first; switching didn't turn out to be the actual fix - the
#     calibrator and exclusion list were - but U8S8 is still the safer,
#     better-supported default and was kept.)
#   - reduce_range=False: reduce_range is an x86-VNNI-less overflow
#     workaround: unnecessary on the ARM (Raspberry Pi) deployment target and
#     only costs precision there.
def quantize_and_eval(model_input_path, model_output_path, excluded_nodes, calibration_reader,
                       val_image_dir, input_name,
                       max_eval_images=200,):
    # out_path = "sweep_candidate.onnx"
    calibration_reader.rewind()  # calibration_reader is stateful/shared across every sweep
    # iteration and the final production quantize_static call below - without
    # this rewind, any call after the reader has been exhausted once raises
    # ort_quant.calibrate.ValueError("No data is collected.").
    ort_quant.quantize_static(
        model_input=model_input_path,
        model_output=model_output_path,
        calibration_data_reader=calibration_reader,
        quant_format=ort_quant.QuantFormat.QDQ,
        calibrate_method=ort_quant.CalibrationMethod.Percentile,
        extra_options={"CalibPercentile": 99.99},
        weight_type=ort_quant.QuantType.QInt8,
        activation_type=ort_quant.QuantType.QUInt8,
        per_channel=True,
        nodes_to_exclude=excluded_nodes,
        reduce_range=False,
    )

    sess = ort.InferenceSession(onnx_static_int8_file_path, providers=["CPUExecutionProvider"])

    # Validation set (val_image_dir) is structured as one directory per
    # ImageNet class (0-999), one image per class - so "how many classes to
    # sample" (max_eval_images) rather than "how many files to glob flatly".
    # One image per class folder; folder name IS the label
    class_dirs = sorted(
        d for d in glob.glob(f"{val_image_dir}/*")
        if os.path.isdir(d)
    )[:max_eval_images]

    correct = 0
    total = 0
    for class_dir in class_dirs:
        label = int(os.path.basename(class_dir))
        
        # BUG FIX: the original glob (commented below) only matched
        # *.jpg/*.JPEG. val_image_dir's files are *.jpeg (lowercase, 4
        # letters), which that pattern silently skipped entirely - every
        # class_dir came back with images=[], so `total` stayed 0 across
        # every sweep iteration and every sweep result was a fake 0.000
        # (falling through to the `return ... if total else 0.0` below),
        # not a real accuracy measurement. Widened to cover common casings.
        extensions = ("*.jpg", "*.JPEG", "*.jpeg", "*.JPG", "*.png", "*.PNG")
        images = list(itertools.chain.from_iterable(
            glob.glob(f"{class_dir}/{ext}") for ext in extensions
        ))
        # images = glob.glob(f"{class_dir}/*.jpg") + glob.glob(f"{class_dir}/*.JPEG")
        if not images:
            continue
        img = Image.open(images[0]).convert("RGB")
        tensor = opencv_preprocess(img).unsqueeze(0).numpy().astype(np.float32)
        logits = sess.run(None, {input_name: tensor})[0]
        pred = int(np.argmax(logits))
        correct += int(pred == label)
        total += 1

    return correct / total if total else 0.0


onnx_static_int8_file_path = "mobilenetv3_small_100_lamb_int8_static_quantization.onnx"
val_image_dir = "./imagenetv2-top-images-format-val"
# Sweep
block_indices = sorted(set(
    n.name.split('/')[2] for n in onnx.load(onnx_prep_file_path).graph.node
    if n.op_type == 'Conv' and n.name.startswith('/blocks/')
))


results = {}
# BLOCK-SENSITIVITY SWEEP (superseded, kept for reference):
# Ran once depthwise convs were re-included wholesale (removing the blanket
# `group > 1` exclusion) crashed accuracy to 6.2%. This loop tested "exclude
# ONLY this one block's nodes, quantize everything else" for each block in
# turn, to find which block was actually responsible rather than guessing.
# Result: blocks.0 excluded alone -> 66.0%; blocks.1 through blocks.5 excluded
# alone -> 3-5% each (barely above the 6.2% nothing-excluded baseline).
# That gap is what proved the corruption originates in blocks.0 specifically
# and propagates forward through every residual Add afterward - blocks.1-5
# looking "bad" here was an artifact of blocks.0 still being broken in those
# runs, not evidence those blocks are themselves fragile. Not re-run after
# blocks.0 was identified since it had already answered the question it was
# built to answer.
# for blk in block_indices:
#     excluded = get_nodes_to_exclude(onnx_prep_file_path, extra_exclude_prefix='blocks.0')
#     acc = quantize_and_eval(onnx_prep_file_path, onnx_static_int8_file_path, excluded, calibration_reader,
#                              val_image_dir, input_tensor.name)
#     results[blk] = acc
#     print(f"{blk}: {acc:.3f}")

# WHOLE-BLOCK EXCLUSION (superseded, kept for reference):
# First fix attempt after the sweep above: exclude every node under
# blocks.0/* entirely. Got accuracy back up to 60.5% - a big recovery, but
# the SQNR debug output (bottom of script) suggested the fragile tensor was
# specifically blocks.0's conv_pw (expansion conv), not the whole block.
# Tested excluding ONLY conv_pw (leaving conv_dw quantized) as a narrower
# fix - that gave 6.2%, identical to the no-exclusion baseline, proving
# conv_pw contributes nothing on its own and conv_dw is the real offender -
# the opposite of what the SQNR tensor name suggested. (The debug tool only
# reports tensors it can name-match against a QDQ boundary; conv_pw appearing
# in that table reflects which tensors were visible to the matcher, not
# necessarily which one is most at fault.)
# excluded = get_nodes_to_exclude(onnx_prep_file_path, extra_exclude_prefix='blocks.0')

# FINAL CHOSEN CONFIG: exclude only blocks.0's conv_dw, re-include conv_pw.
# Gives 61.2% - both a better accuracy AND a smaller exclusion set than the
# whole-block exclusion above (60.5%), since conv_pw now gets quantized too.
excluded = get_nodes_to_exclude(onnx_prep_file_path)  # baseline: conv_pwl, SE, classifier, first conv
excluded += [n.name for n in onnx.load(onnx_prep_file_path).graph.node
             if n.name.startswith('/blocks/blocks.0/') and n.op_type == 'Conv'
             and 'conv_dw' in n.name]

acc = quantize_and_eval(onnx_prep_file_path, onnx_static_int8_file_path, excluded, calibration_reader,
                         val_image_dir, input_tensor.name, max_eval_images=1000)
print(f"blocks.0 conv_dw only excluded: {acc:.3f}")

nodes_to_exclude = get_nodes_to_exclude(onnx_prep_file_path)
print(f"Excluding {len(nodes_to_exclude)} sensitive nodes from quantization.")


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

m = onnx.load(onnx_static_int8_file_path)
print(Counter(n.op_type for n in m.graph.node))


so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = "optimized_check.onnx"
_ = ort.InferenceSession(onnx_static_int8_file_path, so)

m2 = onnx.load("optimized_check.onnx")
print(Counter(n.op_type for n in m2.graph.node))

# 18. Extract label names
# This link shows how to extract the label names from the model:
# https://github.com/huggingface/pytorch-image-models/blob/main/hfdocs/source/models/mobilenet-v3.mdx

url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    f.close()
    
# A residual branch means the block has a shortcut that bypasses some operations and is then combined with the main path.
# MobileNetV3 uses inverted-residual blocks; a residual connection is used when the stride is 1 and input/output channels match.

# An SE (Squeeze-and-Excitation) block computes per-channel scaling factors from the feature map and multiplies them back into the feature map

print("\n\n---- Tensor Image Dump ----")
try:
    # DEBUGGING CONTEXT: this whole block is the preprocessing-mismatch
    # investigation, kept in place (not deleted) since it's the fastest way to
    # re-verify Python/C++ preprocessing agreement if either side ever changes.
    # Using image from class 876 specifically because that is what my C++ logic loaded in the first iteration. Print the label in C++ to see what class-image it loads first and then use that same image here.
    # This step helps verify that the image is preprocessed correctly. 
    img_to_dump_path = "./imagenetv2-top-images-format-val/876/15d0a71957f80a211f1d59bdbee48d89346ae634.jpeg"
    img = Image.open(img_to_dump_path).convert("RGB")
    tensor = preprocess(img)  # PyTorch default transformation
    print(tensor.shape, tensor.mean().item(), tensor.std().item())
    print(tensor.flatten()[:10].tolist())
    tensor.numpy().astype(np.float32).tofile("py_preprocessed.bin")
    # py_preprocessed.bin (PIL/timm path) was the FIRST dump taken and diffed
    # against cpp_preprocessed.bin - that comparison is what originally surfaced
    # the PIL-vs-OpenCV mismatch. Left here so both dumps exist side by side for
    # comparison, though py_opencv_preprocessed.bin below is the one that should
    # actually match C++.


    from PIL import Image
    # IMPORTANT: pass `img` (the already-opened PIL.Image), not img_to_dump_path.
    # Passing the path here would exercise opencv_preprocess's `str` branch
    # (cv2.imread), NOT the `PIL.Image.Image` branch that calibration_reader
    # actually uses via get_next() -> self.transform(image). Verifying the wrong
    # branch gave a false "everything matches" result earlier in debugging;
    # passing `img` here is what actually exercises the real calibration path.
    tensor = opencv_preprocess(img)
    tensor.numpy().astype(np.float32).tofile("py_opencv_preprocessed.bin")

    py_cv = np.fromfile("py_opencv_preprocessed.bin", dtype=np.float32).reshape(3,224,224)
    cpp = np.fromfile("cpp_preprocessed.bin", dtype=np.float32).reshape(3,224,224)  # same C++ dump as before

    diff = py_cv - cpp
    print("mean abs diff:", np.abs(diff).mean())
    print("max abs diff:", np.abs(diff).max())
    print("per-channel mean py_cv:", py_cv.mean(axis=(1,2)), " cpp:", cpp.mean(axis=(1,2)))
    # Expect mean abs diff on the order of 1e-4-1e-3 (float rounding only) once
    # opencv_preprocess matches preprocess.cpp correctly. ~0.026 was the earlier,
    # broken PIL-vs-OpenCV mismatch; ~0.0006 is the confirmed-fixed result.
except Exception as e:
    print(e)
    

print("\n\n---- QDQ Debug Test ----")
# DEBUGGING CONTEXT: this section uses ONNX Runtime's qdq_loss_debug module to
# compare per-tensor activations between the FP32 and INT8 graphs, to
# pinpoint exactly which layers are corrupting accuracy rather than guessing
# from architecture alone. This is what led to identifying conv_pwl, the SE
# convs, and (via the accuracy sweep above) blocks.0's conv_dw as the actual
# fragile layers.
#
# max_images=10 (not the full 1000-image calibration set) is a deliberate
# fix: an earlier version collected activations over the FULL calibration
# set, which kept every intermediate tensor for every image for BOTH the
# FP32 and INT8 models in memory simultaneously (dozens of layers x ~1000
# images x 2 models) - several GB of live Python objects, which crashed VS
# Code's debugger (variable-inspection overhead on top of that already pushed
# it over the edge). 10 images is plenty to rank tensors by SQNR.
#
# NOTE: transform=preprocess (the PIL/timm path) here, not opencv_preprocess.
# This doesn't invalidate the SQNR numbers below - both fp32_activations and
# int8_activations are generated from the SAME debug_reader, so the
# comparison between them is still apples-to-apples - but it does mean these
# debug images are preprocessed slightly differently than what calibration_reader
# used to actually calibrate the quantized model, and differently again from
# what C++ feeds at real inference. Worth switching to opencv_preprocess here
# too if the SQNR numbers ever need to be trusted down to the last dB.
debug_reader = ImageCalibrationDataReader(
    image_dir=CALIBRATION_IMAGE_DIR,
    input_name=input_tensor.name,
    transform=preprocess,
    max_images=10,   # debug only needs a handful of samples
)

fp32_model = "mobilenetv3_small_100_lamb_prep.onnx"
int8_model = "mobilenetv3_small_100_lamb_int8_static_quantization.onnx"

# 1. Add intermediate tensors as outputs
qdq_loss_debug.modify_model_output_intermediate_tensors(
    fp32_model,
    "fp32_debug.onnx"
)

qdq_loss_debug.modify_model_output_intermediate_tensors(
    int8_model,
    "int8_debug.onnx"
)
print("Completed adding intermediate tensors")
# 2. Collect FP32 activations
debug_reader.rewind()

fp32_activations = qdq_loss_debug.collect_activations(
    "fp32_debug.onnx",
    debug_reader
)

print("Collected FP32 Activations")
# 3. Collect INT8/QDQ activations
debug_reader.rewind()

int8_activations = qdq_loss_debug.collect_activations(
    "int8_debug.onnx",
    debug_reader
)

print("Collected INT8 Activations")
# 4. Match QDQ activations against FP32 activations
matches = qdq_loss_debug.create_activation_matching(
    int8_activations,
    fp32_activations
)

print("FP32 activation tensors:", len(fp32_activations))
print("INT8 activation tensors:", len(int8_activations))
print("Matched tensors:", len(matches))

# SQNR RANKING (first draft, superseded by identical logic below - kept for
# reference since it's harmless and shows the original column-header version):
#   qdq_err  = a tensor's own value vs. itself after one quantize->dequantize
#              round-trip (checks: is THIS layer's scale/zero-point sane?)
#   xmodel_err = that tensor's value in the real FP32 forward pass vs. the
#              real INT8 forward pass (checks: has INT8 execution actually
#              diverged from FP32 by the time execution reaches this point?)
# A tensor with healthy qdq_err but poor/negative xmodel_err (as blocks.0's
# descendants showed early on) means the error isn't local to that tensor -
# it was injected upstream and is being carried forward, usually via a
# residual Add. That pattern is what pointed the investigation at blocks.0
# specifically instead of chasing noise accumulation across all layers.
# errors = qdq_loss_debug.compute_activation_error(matches)

# # Sort worst (lowest SQNR = most corrupted) first
# ranked = sorted(errors.items(), key=lambda kv: kv[1].get("xmodel_err", float("inf")))

# print(f"{'Tensor':50s} {'qdq_err (dB)':>14s} {'xmodel_err (dB)':>16s}")
# for name, e in ranked[:20]:
#     print(f"{name:50s} {e['qdq_err']:14.2f} {e.get('xmodel_err', float('nan')):16.2f}")

# Same ranking as the commented block above, just wider top-N (30 vs 20) and
# no header row - this is the version actually left active/run. The two
# blocks are functionally redundant; not consolidated since each run's
# output was being compared against a specific prior debugging turn.
errors = qdq_loss_debug.compute_activation_error(matches)
ranked = sorted(errors.items(), key=lambda kv: kv[1].get("xmodel_err", float("inf")))
for name, e in ranked[:30]:
    print(f"{name:55s} {e['qdq_err']:8.2f} {e.get('xmodel_err', float('nan')):10.2f}")