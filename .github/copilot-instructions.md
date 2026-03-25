# Copilot Instructions for ONNX Runtime Test

This is a Java 21 application for testing ONNX models with NVIDIA GPU acceleration. Designed to load models, run inference, and benchmark performance with optional CUDA optimization.

## Build, Test, and Run Commands

### Build
```bash
./gradlew build          # Full build with tests
./gradlew clean build    # Clean rebuild
```

### Tests
```bash
./gradlew test                    # Run all tests (uses JUnit 5)
./gradlew test --tests PerformanceMonitorTest  # Run single test class
```

### Running the Application
```bash
./gradlew run                # Run main application (OnnxModelTester)
./gradlew runWithGpu         # Run with GPU optimization (custom Gradle task)
```

### Development
```bash
./gradlew dependencies       # View dependency tree
./gradlew build -x test      # Build without running tests
```

## High-Level Architecture

The project consists of three main components:

### 1. Core Model Loading (`OnnxModelLoader`)
- Manages OrtEnvironment and OrtSession (ONNX Runtime API)
- Loads `.onnx` models from disk with optional GPU (CUDA) support
- Provides tensor creation and inference execution
- Handles resource cleanup (sessions, tensors)
- **Key methods**: `loadModel()`, `runInference()`, `createFloatTensor()`

### 2. Performance Monitoring (`PerformanceMonitor`)
- Utility for benchmarking inference latency
- Tracks iterations, total time, average, and throughput
- Outputs statistics via `printSummary()`
- Nanosecond precision timing via `System.nanoTime()`

### 3. Main Entry Point (`OnnxModelTester`)
- Displays system info (CPU cores, JVM memory, Java version)
- Attempts CUDA availability detection
- Serves as a template for custom model tests

### Usage Pattern (from examples)
Most usage follows this lifecycle:
1. Create `OnnxModelLoader` with model path
2. Call `loadModel()` to load the model
3. Create input tensors with `createFloatTensor()`
4. Run inference with `runInference()`
5. Close resources explicitly (tensors, results, loader)

## Key Conventions

### Directory Structure
- `src/main/java/com/onnx/inference/` - Core application code
  - `util/` - Reusable utilities (OnnxModelLoader, PerformanceMonitor)
  - `examples/` - Example implementations for reference
- `src/test/java/` - JUnit 5 tests (mirror main structure)
- `models/` - Directory for `.onnx` model files

### ONNX Runtime Java API
- All ONNX Runtime classes from `ai.onnxruntime.*` package
- Most methods throw `OrtException` - handle consistently
- Resources must be closed: sessions, tensors, results (use try-with-resources or explicit close())
- GPU support: Pass `SessionOptions` to `env.createSession()` (CUDA provider is not explicitly set in current code but infrastructure is ready)

### Testing
- Use JUnit 5 (Jupiter) for all new tests
- Test class naming: `{TargetClass}Test.java` in corresponding test directory
- Run specific tests via `./gradlew test --tests <ClassName>`

### JVM Configuration for GPU Tasks
- Default heap: `-Xms1g -Xmx4g` (configured in `runWithGpu` task)
- Use G1 garbage collector: `-XX:+UseG1GC -XX:MaxGCPauseMillis=200`
- These can be overridden with `Dorg.gradle.jvmargs` property

### Common Input Shapes (Reference from QUICKSTART)
- **Image Classification**: `[batch_size, 3, 224, 224]` (float, RGB 0-1 or -1-1)
- **Object Detection**: `[batch_size, 3, height, width]` (float)
- **NLP/BERT**: `[batch_size, sequence_length]` (int64 token IDs)
- **Segmentation**: `[batch_size, 3, height, width]` → `[batch_size, num_classes, height, width]`

### GPU/CUDA Support
- `OnnxModelLoader` now attempts to enable GPU (CUDA) execution provider on model load
- Falls back gracefully to CPU if GPU unavailable
- Logs which execution provider is being used (check console output on model load)
- To enable GPU: Install GPU-enabled ONNX Runtime package + NVIDIA CUDA 11.8+ + cuDNN 8.5+
- Without GPU, inference still works on CPU (slower but functional)

### Dependencies
- ONNX Runtime: 1.18.0 (with CPU+GPU support)
- Logging: SLF4J 2.0.9 with Logback 1.4.11 (configure via `logback.xml`)
- Testing: JUnit 5 (Jupiter) with JUnit 4 legacy support

## Common Patterns

### Model Inference Loop
```java
OnnxModelLoader loader = new OnnxModelLoader("path/to/model.onnx");
loader.loadModel();
try {
    for (Data data : dataset) {
        OnnxTensor input = loader.createFloatTensor(data.floatArray, data.shape);
        Map<String, OnnxTensor> inputs = Map.of("input_name", input);
        OrtSession.Result results = loader.runInference(inputs);
        // Process results
        input.close();
        results.close();
    }
} finally {
    loader.close();
}
```

### Benchmarking
```java
PerformanceMonitor monitor = new PerformanceMonitor("Model Name");
for (int i = 0; i < iterations; i++) {
    monitor.start();
    loader.runInference(inputs);
    monitor.end();
}
monitor.printSummary();
```

## Troubleshooting Reference
- **GPU not detected**: Verify CUDA Toolkit 11.8+ and cuDNN 8.5+ installed; check `nvidia-smi` and `nvcc --version`
- **OutOfMemoryError**: Reduce batch size or increase JVM heap with `-Xmx` parameter
- **Model loading fails**: Verify `.onnx` file path and ONNX Runtime version compatibility
- **Library not found** (e.g., libcudnn): Ensure CUDA/cuDNN paths in system PATH and LD_LIBRARY_PATH
