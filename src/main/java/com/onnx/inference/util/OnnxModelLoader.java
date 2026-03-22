package com.onnx.inference.util;

import ai.onnxruntime.*;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
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
        SessionOptions opts = new SessionOptions();
        
        // Enable CUDA execution provider for GPU acceleration
        opts.addCudaExecutionProvider();
        
        // Fallback to CPU if CUDA not available
        opts.addCpuExecutionProvider();
        
        // Set optimization level
        opts.setExecutionMode(ExecutionMode.SEQUENTIAL);
        
        // Configure graph optimization
        opts.setOptimizationLevel(GraphOptimizationLevel.ALL_OPTIMIZED);
        
        System.out.println("Loading model from: " + modelPath);
        this.session = env.createSession(modelPath, opts);
        
        printModelInfo();
    }
    
    /**
     * Print model input/output information
     */
    private void printModelInfo() throws OrtException {
        System.out.println("\n--- Model Information ---");
        System.out.println("Input nodes:");
        for (NodeArg input : session.getInputInfo().values()) {
            System.out.println("  - " + input.getName() + ": " + Arrays.toString(input.getShape()));
        }
        
        System.out.println("Output nodes:");
        for (NodeArg output : session.getOutputInfo().values()) {
            System.out.println("  - " + output.getName() + ": " + Arrays.toString(output.getShape()));
        }
    }
    
    /**
     * Run inference on input data
     */
    public OrtSession.Result runInference(Map<String, OnnxTensor> inputs) throws OrtException {
        if (session == null) {
            throw new OrtException("Model not loaded. Call loadModel() first.");
        }
        
        List<String> outputNames = new ArrayList<>(session.getOutputInfo().keySet());
        return session.run(inputs, outputNames);
    }
    
    /**
     * Create input tensor from float array
     */
    public OnnxTensor createFloatTensor(float[] data, long[] shape) throws OrtException {
        return OnnxTensor.createTensor(env, data, shape);
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
