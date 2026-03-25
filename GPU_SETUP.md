# GPU Setup Troubleshooting Guide

## Current Status
- ✓ CUDA 13.2 installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`
- ✓ cuDNN v9.20 installed at `C:\Program Files\NVIDIA\CUDNN\v9.20`
- ✓ ONNX Runtime 1.20.0 GPU package in Maven
- ✓ Java library paths configured in build.gradle
- ⚠ GPU execution provider not detected at runtime

## The Issue

The Maven `onnxruntime_gpu` package may not include pre-built GPU execution provider libraries for all CUDA versions. The package structure suggests it's primarily for CPU-only execution.

## Solutions

### Option 1: Use Pre-built ONNX Runtime GPU Binaries (Recommended)
1. Download pre-built ONNX Runtime GPU JAR from GitHub releases:
   - https://github.com/microsoft/onnxruntime/releases
   - Look for: `onnxruntime-java` with GPU support
   - Version 1.20.0 or later

2. Replace the Maven dependency with a local JAR:
   ```gradle
   implementation files('libs/onnxruntime-java-gpu.jar')
   ```

3. Place downloaded JAR in `libs/` folder

### Option 2: Build from Source
1. Clone ONNX Runtime repository:
   ```bash
   git clone https://github.com/microsoft/onnxruntime.git
   cd onnxruntime
   ```

2. Build with CUDA 13.2 support following:
   - https://github.com/microsoft/onnxruntime/blob/main/BUILD.md
   - https://github.com/microsoft/onnxruntime/tree/main/java

3. Use locally built JAR in your project

### Option 3: Fallback to PyORT or C++ Wrapper
If Java integration is flexible, consider using Python with ONNX Runtime:
```bash
pip install onnxruntime-gpu
```

Then call from Java via `ProcessBuilder` or JNI wrapper

## Verification

Once GPU is properly configured, output will show:
```
GPU (CUDA) execution provider enabled
Average time: ~0.05 ms (much faster than CPU 0.15ms)
Throughput: ~20000+ inferences/sec
```

## Current Workaround

The system currently works well on **CPU**:
- Average time: 0.150 ms per inference
- Throughput: 6650+ inferences/sec
- MNIST model size: Only 26 KB (small enough that CPU is adequate)

For larger models requiring GPU acceleration, implement one of the solutions above.

## References
- ONNX Runtime Java: https://github.com/microsoft/onnxruntime/tree/main/java
- ONNX Runtime GPU: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- CUDA 13.2 Documentation: https://docs.nvidia.com/cuda/
- cuDNN v9.20: https://docs.nvidia.com/deeplearning/cudnn/reference/
