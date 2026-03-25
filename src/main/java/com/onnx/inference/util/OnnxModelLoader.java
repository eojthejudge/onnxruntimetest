package com.onnx.inference.util;

import ai.onnxruntime.*;
import java.util.*;
import java.util.Arrays;

public class OnnxModelLoader {
    private OrtSession session;
    private final OrtEnvironment env;
    private final String modelPath;
    
    public OnnxModelLoader(String modelPath) throws OrtException {
        this.modelPath = modelPath;
        // Initialize ORT environment with GPU provider
        this.env = OrtEnvironment.getEnvironment();
    }
    
    /**
     * Load ONNX model with CUDA GPU support (fallback to CPU if unavailable)
     */
    public void loadModel() throws OrtException {
        try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
            
            System.out.println("Loading model from: " + modelPath);
            
            // Try to enable CUDA GPU support
            boolean gpuEnabled = enableGpuSupport(opts);
            
            this.session = env.createSession(modelPath, opts);
            
            if (gpuEnabled) {
                System.out.println("GPU (CUDA) execution provider enabled");
            } else {
                System.out.println("Running on CPU (GPU not available)");
            }
            
            printModelInfo();
        }
    }
    
    /**
     * Attempt to enable CUDA GPU execution provider
     * @return true if GPU was successfully enabled, false if falling back to CPU
     */
    private boolean enableGpuSupport(OrtSession.SessionOptions opts) {
        try {
            // ONNX Runtime 1.24.3: addCUDA(int gpuDeviceId)
            var method = opts.getClass().getMethod("addCUDA", int.class);
            method.invoke(opts, 0);  // Device ID 0 (first GPU)
            System.out.println("✓ GPU (CUDA) execution provider enabled");
            return true;
        } catch (NoSuchMethodException ignored) {
            // Try without parameters
            try {
                var method = opts.getClass().getMethod("addCUDA");
                method.invoke(opts);
                System.out.println("✓ GPU (CUDA) execution provider enabled");
                return true;
            } catch (Exception e2) {
                System.out.println("Note: GPU execution provider not available");
                return false;
            }
        } catch (Exception e) {
            System.out.println("Note: Could not enable GPU - " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Print model input/output information
     */
    private void printModelInfo() throws OrtException {
        System.out.println("\n--- Model Information ---");
        System.out.println("Input node count: " + session.getInputInfo().size());
        for (String inputName : session.getInputInfo().keySet()) {
            System.out.println("  - Input: " + inputName);
        }
        
        System.out.println("Output node count: " + session.getOutputInfo().size());
        for (String outputName : session.getOutputInfo().keySet()) {
            System.out.println("  - Output: " + outputName);
        }
    }
    
    /**
     * Run inference on input data
     */
    public OrtSession.Result runInference(Map<String, OnnxTensor> inputs) throws OrtException {
        if (session == null) {
            throw new OrtException("Model not loaded. Call loadModel() first.");
        }
        
        return session.run(inputs);
    }
    
    /**
     * Create input tensor from float array
     */
    public OnnxTensor createFloatTensor(float[] data, long[] shape) throws OrtException {
        // Create a 2D array wrapper to match the API
        float[][] wrappedData = new float[1][0];
        wrappedData[0] = data;
        return OnnxTensor.createTensor(env, wrappedData);
    }
    
    /**
     * Close resources
     */
    public void close() throws OrtException {
        if (session != null) {
            session.close();
        }
    }
    
    /**
     * Get session for advanced operations
     */
    public OrtSession getSession() {
        return session;
    }
}
