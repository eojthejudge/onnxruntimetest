# ONNX Models Directory

This directory contains ONNX model files for testing and inference.

## Included Models

### mnist-12.onnx
- **Description**: MNIST handwritten digit classifier
- **Source**: [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)
- **Input**: `[batch_size, 1, 28, 28]` - grayscale images (normalized to [0, 1])
- **Output**: `[batch_size, 10]` - probabilities for digits 0-9
- **Size**: ~26 KB
- **Example Usage**: Run `./gradlew runMnist`

## Adding Your Own Models

1. Download or export your ONNX model (`.onnx` file)
2. Place it in this directory
3. Create a test class in `src/main/java/com/onnx/inference/examples/` following the pattern of `MnistTester.java`
4. Update your test to use the correct input/output node names and shapes

### Common Model Sources
- [ONNX Model Zoo](https://github.com/onnx/models)
- [Hugging Face ONNX Models](https://huggingface.co/models?library=onnx)
- [PyTorch to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [TensorFlow to ONNX](https://github.com/onnx/tensorflow-onnx)

## Model Testing Workflow

1. **Understand the model structure**:
   ```bash
   ./gradlew run  # View model information printed to console
   ```

2. **Create test data**:
   - Generate synthetic inputs matching the model's input shape
   - Or load real data (images, text, etc.)

3. **Benchmark performance**:
   ```bash
   ./gradlew runWithGpu
   ```

## Notes
- Large models (>100 MB) should be stored separately or added to `.gitignore`
- The `.onnx` extension is already ignored in `.gitignore` (see line 30)
- For GPU inference, ensure NVIDIA CUDA Toolkit and cuDNN are installed
