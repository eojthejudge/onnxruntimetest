package com.onnx.inference;

public class OnnxModelTester {
    public static void main(String[] args) {
        System.out.println("=== ONNX Model Tester with GPU Acceleration ===");
        
        // Display GPU information
        displayGpuInfo();
        
        // Example model testing
        try {
            OnnxModelTester tester = new OnnxModelTester();
            System.out.println("\nReady to test ONNX models with GPU acceleration.");
            System.out.println("Usage: Place your .onnx model files in the 'models' directory.");
            System.out.println("Then update this class to load and test your specific model.");
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void displayGpuInfo() {
        System.out.println("\n--- GPU/System Information ---");
        System.out.println("Available processors: " + Runtime.getRuntime().availableProcessors());
        System.out.println("Max memory: " + (Runtime.getRuntime().maxMemory() / 1024 / 1024) + " MB");
        System.out.println("Java version: " + System.getProperty("java.version"));
        System.out.println("OS: " + System.getProperty("os.name") + " (" + System.getProperty("os.arch") + ")");
        
        // Try to detect CUDA
        try {
            checkCudaAvailability();
        } catch (Exception e) {
            System.out.println("CUDA check: Not available or not properly configured");
        }
    }
    
    private static void checkCudaAvailability() {
        String osName = System.getProperty("os.name").toLowerCase();
        String javaLibraryPath = System.getProperty("java.library.path");
        
        System.out.println("Java library path: " + javaLibraryPath);
        
        // Note: More sophisticated CUDA detection would require native libraries
        System.out.println("Note: For full GPU support, ensure CUDA Toolkit and cuDNN are installed.");
    }
}
