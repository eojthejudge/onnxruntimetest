package com.onnx.inference.examples;

import com.onnx.inference.util.OnnxModelLoader;
import com.onnx.inference.util.PerformanceMonitor;
import ai.onnxruntime.*;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Example: Testing MNIST digit classification model
 * 
 * MNIST input: [batch_size, 1, 28, 28] - grayscale 28x28 images
 * Output: [batch_size, 10] - probabilities for digits 0-9
 */
public class MnistTester {
    public static void main(String[] args) throws OrtException {
        String modelPath = "models/mnist-12.onnx";
        
        System.out.println("=== MNIST Digit Classification Example ===\n");
        
        OnnxModelLoader loader = new OnnxModelLoader(modelPath);
        loader.loadModel();
        
        testMnistInference(loader);
        
        loader.close();
    }
    
    private static void testMnistInference(OnnxModelLoader loader) throws OrtException {
        System.out.println("Testing MNIST inference with random digit images...\n");
        
        // Create synthetic MNIST-like input: [1, 1, 28, 28] (batch size 1)
        float[] imageData = generateRandomDigitImage();
        
        // Get the input node name
        String inputName = loader.getSession().getInputInfo().keySet().iterator().next();
        System.out.println("Input node: " + inputName);
        
        // Create 4D tensor [batch=1, channels=1, height=28, width=28]
        float[][][][] tensor4D = new float[1][1][28][28];
        int idx = 0;
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                tensor4D[0][0][i][j] = imageData[idx++];
            }
        }
        
        OnnxTensor inputTensor = OnnxTensor.createTensor(
            OrtEnvironment.getEnvironment(),
            tensor4D
        );
        
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(inputName, inputTensor);
        
        // Run inference
        try (OrtSession.Result results = loader.runInference(inputs)) {
            Object output = results.get(0).getValue();
            
            if (output instanceof float[][]) {
                float[][] predictions = (float[][]) output;
                printPredictions(predictions[0]);
            } else if (output instanceof float[]) {
                printPredictions((float[]) output);
            }
        }
        
        inputTensor.close();
        
        // Performance benchmark
        System.out.println("\n--- Performance Benchmark (100 iterations) ---");
        benchmarkInference(loader, imageData, inputName);
    }
    
    private static void benchmarkInference(OnnxModelLoader loader, float[] imageData, String inputName) throws OrtException {
        PerformanceMonitor monitor = new PerformanceMonitor("MNIST Inference");
        
        // Pre-build the 4D tensor shape
        float[][][][] tensor4D = new float[1][1][28][28];
        
        for (int i = 0; i < 100; i++) {
            monitor.start();
            
            try {
                // Reshape data to 4D tensor
                int idx = 0;
                for (int x = 0; x < 28; x++) {
                    for (int y = 0; y < 28; y++) {
                        tensor4D[0][0][x][y] = imageData[idx++];
                    }
                }
                
                OnnxTensor inputTensor = OnnxTensor.createTensor(
                    OrtEnvironment.getEnvironment(),
                    tensor4D
                );
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put(inputName, inputTensor);
                
                try (OrtSession.Result results = loader.runInference(inputs)) {
                    // Just consume the results
                }
                
                inputTensor.close();
            } catch (OrtException e) {
                System.err.println("Error during benchmark iteration " + i + ": " + e.getMessage());
            }
            
            monitor.end();
        }
        
        monitor.printSummary();
    }
    
    private static void printPredictions(float[] output) {
        System.out.println("\nDigit Predictions (output probabilities):");
        
        float maxProb = Float.NEGATIVE_INFINITY;
        int predictedDigit = -1;
        
        for (int i = 0; i < output.length; i++) {
            System.out.printf("  Digit %d: %.4f%n", i, output[i]);
            if (output[i] > maxProb) {
                maxProb = output[i];
                predictedDigit = i;
            }
        }
        
        System.out.println("\nPredicted digit: " + predictedDigit + " (confidence: " + String.format("%.2f%%", maxProb * 100) + ")");
    }
    
    private static float[] generateRandomDigitImage() {
        // Generate a random 28x28 grayscale image [0, 1]
        Random rand = new Random(System.currentTimeMillis());
        float[] image = new float[28 * 28];
        
        for (int i = 0; i < image.length; i++) {
            image[i] = rand.nextFloat(); // Random value between 0 and 1
        }
        
        return image;
    }
}
