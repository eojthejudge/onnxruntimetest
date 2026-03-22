# ONNX Model Tester with GPU Acceleration

A Java 21 application for testing ONNX models with NVIDIA GPU acceleration (CUDA).

## Prerequisites

### System Requirements
- **Java 21** or later
- **NVIDIA CUDA Toolkit 11.8+** (for GPU support)
- **cuDNN 8.5+** (for GPU acceleration)
- **Gradle 8.0+** (or use included wrapper)

### CUDA Setup (Windows)
1. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
2. Install cuDNN from https://developer.nvidia.com/cudnn
3. Add to system PATH:
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib`
4. Verify installation: `nvcc --version`

### GPU Verification
```bash
# Check available GPU
nvidia-smi

# Check if CUDA is accessible
nvcc --version
```

## Project Structure

```
├── build.gradle              # Gradle configuration with ONNX Runtime GPU dependencies
├── settings.gradle           # Gradle settings
├── gradle/                   # Gradle wrapper
├── src/
│   ├── main/java/
│   │   └── com/onnx/inference/
│   │       ├── OnnxModelTester.java      # Main entry point
│   │       ├── util/
│   │       │   ├── OnnxModelLoader.java  # Model loading with GPU support
│   │       │   └── PerformanceMonitor.java # Performance benchmarking
│   │       └── examples/
│   │           └── SampleModelTester.java # Example usage
│   └── test/java/            # Unit tests
└── models/                   # Place your .onnx models here
```

## Building the Project

### Build
```bash
./gradlew build
```

### Run with GPU Optimization
```bash
./gradlew runWithGpu
```

### Run standard
```bash
./gradlew run
```

### Run tests
```bash
./gradlew test
```

## Usage Examples

### Basic Model Testing
```java
import com.onnx.inference.util.OnnxModelLoader;
import ai.onnxruntime.*;

try {
    OnnxModelLoader loader = new OnnxModelLoader("path/to/model.onnx");
    loader.loadModel();
    
    // Create input tensor
    float[] inputData = new float[...];
    long[] inputShape = {...};
    OnnxTensor inputTensor = loader.createFloatTensor(inputData, inputShape);
    
    Map<String, OnnxTensor> inputs = new HashMap<>();
    inputs.put("input_name", inputTensor);
    
    // Run inference
    OrtSession.Result results = loader.runInference(inputs);
    
    // Process results...
    
    inputTensor.close();
    results.close();
    loader.close();
} catch (OrtException e) {
    e.printStackTrace();
}
```

### Performance Benchmarking
```java
import com.onnx.inference.util.PerformanceMonitor;

PerformanceMonitor monitor = new PerformanceMonitor("My Model");

for (int i = 0; i < 100; i++) {
    monitor.start();
    // Run inference...
    monitor.end();
}

monitor.printSummary();
```

## GPU Acceleration Features

- **CUDA EP (Execution Provider)**: Offloads inference to NVIDIA GPU
- **Graph Optimization**: Automatically optimizes model graph
- **Memory Management**: Efficient GPU memory allocation
- **Fallback**: Automatically falls back to CPU if GPU unavailable

## Performance Tuning

### JVM Arguments (Modified in build.gradle)
```gradle
jvmArgs = [
    '-XX:+UseG1GC',           # Use G1 garbage collector
    '-XX:MaxGCPauseMillis=200',
    '-Xms1g',                  # Initial heap size
    '-Xmx4g'                   # Maximum heap size
]
```

### ONNX Runtime Options
```java
SessionOptions opts = new SessionOptions();
opts.addCudaExecutionProvider();
opts.setOptimizationLevel(GraphOptimizationLevel.ALL_OPTIMIZED);
```

## Troubleshooting

### GPU Not Detected
1. Verify CUDA installation: `nvcc --version`
2. Check GPU availability: `nvidia-smi`
3. Ensure CUDA paths in system environment
4. Try CPU fallback (application will still work)

### OutOfMemoryError
- Reduce batch size
- Adjust JVM heap size (-Xmx parameter)
- Check GPU memory with `nvidia-smi`

### Model Loading Issues
- Verify .onnx file path
- Check model compatibility with ONNX Runtime version
- Review model input/output specifications

## Dependencies

- **ONNX Runtime**: 1.18.0 (with GPU support)
- **SLF4J & Logback**: Logging framework
- **JUnit 5**: Testing framework
- **OpenCV** (optional): For image processing tasks

## Resources

- [ONNX Runtime Java API](https://github.com/microsoft/onnxruntime/tree/main/java)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [ONNX Model Zoo](https://github.com/onnx/models)

## License

This project is open-source. See LICENSE file for details.
