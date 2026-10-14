# Hardware Acceleration and Future FPGA Deployment Architecture

This document outlines the hardware acceleration strategy for the AI Medicine Identification and Prescription Verification System, covering current GPU/CPU optimizations and the architectural pathway for future edge FPGA synthesis.

---

## 1. Current Hardware Architecture

The system features dynamic runtime device detection (`src/config.py`):

```python
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
```

### Optimizations implemented:
- **Batched CNN Inference:** Reference image feature extraction processes images in configurable batches (default `batch_size=32`), minimizing host-to-device memory transfer overhead.
- **Persistent Embedding Caching:** All 1,280-dimensional MobileNetV2 embeddings for reference packaging images are computed once and stored in an indexed NumPy archive (`cache/reference_embeddings.npz`), avoiding redundant neural network passes.
- **Indexed Full-Text Search (FTS5):** Catalogue lookups against the 253,973-record pharmaceutical catalogue are handled via SQLite FTS5 in C-level memory, avoiding Python overhead and eliminating brute-force scanning.
- **Graceful Fallback:** If CUDA is unavailable (e.g. standard laptop CPU), the entire pipeline executes on CPU without raising unhandled exceptions or degrading verification accuracy.

---

## 2. FPGA Deployment Roadmap

Field-Programmable Gate Arrays (FPGAs) offer high energy efficiency, deterministic ultra-low latency, and real-time inference suitable for embedded clinical dispensing kiosks and point-of-care mobile scanners.

### Target FPGA Hardware Families
- **AMD Xilinx Kria KV260 / KR260:** Powered by Zynq UltraScale+ MPSoC with Deep Learning Processing Unit (DPUczdx8g).
- **AMD Xilinx Versal AI Edge:** Dedicated AI Engines (AIE-ML) optimized for tensor operations.
- **Intel Agilex / Cyclone V SoC:** Using Intel OpenVINO / FPGA AI Suite.

---

## 3. Conversion and Quantization Pipeline

To deploy the vision backbone on an FPGA, the PyTorch model must undergo the following sequential stages:

```
[PyTorch MobileNetV2] 
         │
         ▼ (1. Export)
    [ONNX Model] 
         │
         ▼ (2. Validation & Simplification)
[ONNX Runtime Validation]
         │
         ▼ (3. INT8 Calibration & Quantization)
[Quantized ONNX / Vitis AI Quantizer]
         │
         ▼ (4. Compilation for Target DPU)
[Xilinx .xmodel / FPGA Bitstream]
```

### Stage 1: PyTorch to ONNX Export
Export the feature extractor (with classification head removed) to standard ONNX (`opset_version=17`):

```python
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier = torch.nn.Identity()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
torch.onnx.export(
    model,
    dummy_input,
    "mobilenet_v2_features.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input_tensor"],
    output_names=["feature_embedding"],
    dynamic_axes={"input_tensor": {0: "batch_size"}, "feature_embedding": {0: "batch_size"}}
)
```

### Stage 2: ONNX Runtime Validation
Verify that numerical outputs between PyTorch and ONNX match within tolerance ($\epsilon < 10^{-5}$):

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("mobilenet_v2_features.onnx", providers=["CPUExecutionProvider"])
ort_inputs = {"input_tensor": dummy_input.numpy()}
ort_outputs = session.run(None, ort_inputs)

np.testing.assert_allclose(model(dummy_input).detach().numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
print("ONNX numerical validation passed!")
```

### Stage 3: INT8 Post-Training Quantization (PTQ)
FPGAs achieve maximum throughput when weights and activations are quantized to INT8 fixed-point arithmetic:
- **Calibration Dataset:** 200–500 unlabelled medicine package images are passed through the quantizer to determine optimal scaling factors and zero-points.
- **Toolchain:** `vai_q_onnx` (Vitis AI ONNX Quantizer) or `vai_q_pytorch`.
- Symmetrical quantization with power-of-two scaling is preferred for DSP slice mapping.

### Stage 4: DPU Compilation and Hardware Synthesis
1. Compile the quantized model using the Vitis AI Compiler (`vaic`):
   ```bash
   vai_c_onnx -m mobilenet_v2_quantized.onnx -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o ./fpga_build -n mobilenet_v2_kv260
   ```
2. The compiler generates a hardware ELF file (`mobilenet_v2_kv260.xmodel`) containing instructions for the FPGA DPU processing core.
3. Deploy the compiled model onto the FPGA board via Vitis AI Runtime (VART) or Python bindings.

---

## 4. Hardware Limitations and Engineering Disclaimers

> [!IMPORTANT]
> **FPGA Deployment Verification Notice:**
> - Exporting to ONNX is fully supported in software.
> - Actual FPGA synthesis, bitstream generation, and hardware deployment **cannot be validated without access to target FPGA hardware and vendor-specific toolchains** (AMD Vivado, Vitis AI, or Intel Quartus Prime).
> - We do not claim completed FPGA synthesis until the bitstream has been flashed and timed on verified physical hardware.
