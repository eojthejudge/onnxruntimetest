# Quick Start Guide

## 1. Prerequisites Check

Before running the application, ensure you have:

```bash
# Check Java version (should be 21+)
java -version

# Check Gradle (if not using wrapper)
gradle --version

# Check GPU availability
nvidia-smi
```

## 2. Build the Project

```bash
# Windows
gradlew.bat build

# Linux/macOS
./gradlew build
```

## 3. Prepare Your Model

1. Obtain an ONNX model (.onnx file)
2. Place it in the `models/` directory
3. Note the model's input/output specifications

## 4. Create Your Test

Edit `src/main/java/com/onnx/inference/OnnxModelTester.java` or create a new test class:

```java
import com.onnx.inference.util.OnnxModelLoader;
import com.onnx.inference.util.PerformanceMonitor;

public class MyModelTest {
    public static void main(String[] args) throws Exception {
        // Load model
        OnnxModelLoader loader = new OnnxModelLoader("models/my-model.onnx");
        loader.loadModel();
        
        // Prepare input
        float[] inputData = new float[...]; // Your input data
        long[] inputShape = {...}; // Your input shape
        
        // Create tensor and run inference
        var inputTensor = loader.createFloatTensor(inputData, inputShape);
        var inputs = Map.of("input", inputTensor);
        var results = loader.runInference(inputs);
        
        // Process results
        Object output = results.get(0).getValue();
        System.out.println("Output: " + output);
        
        // Cleanup
        inputTensor.close();
        results.close();
        loader.close();
    }
}
```

## 5. Run with GPU

```bash
# Run with GPU optimization
gradlew runWithGpu

# Or run specific class
gradlew run --args="com.onnx.inference.OnnxModelTester"
```

## 6. Performance Monitoring

Use the `PerformanceMonitor` class for benchmarking:

```java
PerformanceMonitor monitor = new PerformanceMonitor("Model Test");

for (int i = 0; i < 100; i++) {
    monitor.start();
    // Run inference
    monitor.end();
}

monitor.printSummary();
// Output:
// --- Performance Summary: Model Test ---
// Iterations: 100
// Average time: 23.456 ms
// Throughput: 42.63 inferences/sec
```

## 7. Common Input Shapes

### Image Classification (ResNet, VGG, etc.)
- Input: `[batch_size, 3, 224, 224]` (float)
- Format: RGB, normalized to [0, 1] or [-1, 1]

### Object Detection (YOLO, SSD, etc.)
- Input: `[batch_size, 3, height, width]` (float)
- Output: Bounding boxes, scores, class IDs

### NLP (BERT, GPT, etc.)
- Input: `[batch_size, sequence_length]` (int64)
- Typically token IDs and attention masks

### Segmentation
- Input: `[batch_size, 3, height, width]` (float)
- Output: `[batch_size, num_classes, height, width]` (float)

## 8. Troubleshooting

### Model loads but slow (using CPU instead of GPU)
1. Verify GPU detection: Check console output during load
2. Check CUDA Path: Ensure `java.library.path` includes CUDA libraries
3. Verify CUDA installation: Run `deviceQuery` from CUDA samples

### OutOfMemoryError
- Reduce batch size
- Increase JVM heap: Edit `build.gradle` and set `-Xmx8g`

### "libcudnn.so not found"
- Install cuDNN
- Add to LD_LIBRARY_PATH (Linux) or PATH (Windows)

### Model architecture not supported
- Check ONNX Runtime compatibility matrix
- Try converting model with ONNX Optimizer or ONNX Simplifier

## 9. Next Steps

- Explore example models from [ONNX Model Zoo](https://github.com/onnx/models)
- Customize `OnnxModelLoader` for your specific needs
- Implement batch inference for better throughput
- Add pre-processing/post-processing pipelines

## 10. Useful Commands

```bash
# Clean build
gradlew clean build

# Run tests
gradlew test

# Generate distributions
gradlew distributions

# View dependencies
gradlew dependencies

# Run with custom JVM args
gradlew run -Dorg.gradle.jvmargs="-Xmx8g"
```

## Resources

- [ONNX Runtime Docs](https://onnxruntime.ai/)
- [ONNX Zoo](https://github.com/onnx/models)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [GitHub - ONNX Runtime Java](https://github.com/microsoft/onnxruntime/tree/main/java)
