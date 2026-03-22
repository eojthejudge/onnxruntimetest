package com.onnx.inference.util;

import ai.onnxruntime.*;
import java.util.*;

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
     * Load ONNX model with CUDA GPU support
     */
    public void loadModel() throws OrtException {
        try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
            
            System.out.println("Loading model from: " + modelPath);
            this.session = env.createSession(modelPath, opts);
            
            printModelInfo();
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
