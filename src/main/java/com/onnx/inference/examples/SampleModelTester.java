package com.onnx.inference.examples;

import com.onnx.inference.util.OnnxModelLoader;
import com.onnx.inference.util.PerformanceMonitor;
import ai.onnxruntime.*;
import java.util.*;

/**
 * Example model tester - customize for your specific model
 */
public class SampleModelTester {
    
    /**
     * Example: Test a simple image classification model
     * Modify this to match your model's input/output specifications
     */
    public static void testImageClassificationModel(String modelPath) throws OrtException {
        OnnxModelLoader loader = new OnnxModelLoader(modelPath);
        PerformanceMonitor monitor = new PerformanceMonitor("Image Classification");
        
        try {
            // Load the model
            loader.loadModel();
            
            // Create sample input (3x224x224 for typical image models)
            float[] inputData = generateRandomInput(1 * 3 * 224 * 224);
            long[] inputShape = {1, 3, 224, 224};
            
            // Run inference multiple times for benchmarking
            int iterations = 10;
            System.out.println("\nRunning " + iterations + " inference iterations...");
            
            for (int i = 0; i < iterations; i++) {
                monitor.start();
                
                // Create input tensor
                OnnxTensor inputTensor = loader.createFloatTensor(inputData, inputShape);
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("images", inputTensor);
                
                // Run inference
                OrtSession.Result results = loader.runInference(inputs);
                
                // Print first iteration results
                if (i == 0) {
                    printResults(results);
                }
                
                // Clean up
                inputTensor.close();
                results.close();
                
                monitor.end();
                
                if ((i + 1) % 5 == 0) {
                    System.out.println("  Completed: " + (i + 1) + "/" + iterations);
                }
            }
            
            monitor.printSummary();
            
        } finally {
            loader.close();
        }
    }
    
    /**
     * Example: Test a general ONNX model with custom input
     */
    public static void testGenericModel(String modelPath, float[] inputData, long[] inputShape) throws OrtException {
        OnnxModelLoader loader = new OnnxModelLoader(modelPath);
        PerformanceMonitor monitor = new PerformanceMonitor("Generic Model Test");
        
        try {
            loader.loadModel();
            monitor.start();
            
            // Create input tensor
            OnnxTensor inputTensor = loader.createFloatTensor(inputData, inputShape);
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(loader.getSession().getInputInfo().keySet().iterator().next(), inputTensor);
            
            // Run inference
            OrtSession.Result results = loader.runInference(inputs);
            
            monitor.end();
            monitor.printSummary();
            
            // Print results
            printResults(results);
            
            // Clean up
            inputTensor.close();
            results.close();
            
        } finally {
            loader.close();
        }
    }
    
    /**
     * Print inference results
     */
    private static void printResults(OrtSession.Result results) throws OrtException {
        System.out.println("\n--- Inference Results ---");
        for (Map.Entry<String, OnnxValue> entry : results) {
            String outputName = entry.getKey();
            Object output = entry.getValue().getValue();
            
            System.out.println("Output '" + outputName + "':");
            if (output instanceof float[][][]) {
                float[][][] data = (float[][][]) output;
                System.out.println("  Shape: [" + data.length + "][" + data[0].length + "][" + data[0][0].length + "]");
                if (data.length > 0 && data[0].length > 0 && data[0][0].length > 0) {
                    System.out.println("  Sample values: " + Arrays.toString(Arrays.copyOfRange(data[0][0], 0, Math.min(5, data[0][0].length))));
                }
            } else if (output instanceof float[]) {
                float[] data = (float[]) output;
                System.out.println("  Shape: [" + data.length + "]");
                System.out.println("  Sample values: " + Arrays.toString(Arrays.copyOfRange(data, 0, Math.min(5, data.length))));
            } else if (output instanceof long[]) {
                long[] data = (long[]) output;
                System.out.println("  Shape: [" + data.length + "] (int64)");
                System.out.println("  Sample values: " + Arrays.toString(Arrays.copyOfRange(data, 0, Math.min(5, data.length))));
            } else {
                System.out.println("  Type: " + output.getClass().getSimpleName());
            }
        }
    }
    
    /**
     * Generate random input for testing
     */
    private static float[] generateRandomInput(int size) {
        float[] data = new float[size];
        Random rand = new Random(42); // Fixed seed for reproducibility
        for (int i = 0; i < size; i++) {
            data[i] = rand.nextFloat();
        }
        return data;
    }
}
