package com.onnx.inference.examples;

import com.onnx.inference.util.OnnxModelLoader;
import com.onnx.inference.util.PerformanceMonitor;
import ai.onnxruntime.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.*;

/**
 * Advanced model tester with support for different model types and configurations
 */
public class AdvancedModelTester {
    private static final Logger logger = LoggerFactory.getLogger(AdvancedModelTester.class);
    
    /**
     * Test detection model (YOLO, Faster R-CNN, etc.)
     */
    public static void testDetectionModel(String modelPath, float[] imageData, long[] imageShape) 
            throws OrtException {
        logger.info("Testing object detection model: {}", modelPath);
        
        OnnxModelLoader loader = new OnnxModelLoader(modelPath);
        PerformanceMonitor monitor = new PerformanceMonitor("Object Detection");
        
        try {
            loader.loadModel();
            
            // Warm-up runs (GPU initialization)
            logger.info("Warming up GPU...");
            for (int i = 0; i < 3; i++) {
                runInferenceOnce(loader, imageData, imageShape);
            }
            
            // Benchmark runs
            int iterations = 20;
            for (int i = 0; i < iterations; i++) {
                monitor.start();
                OrtSession.Result results = runInferenceOnce(loader, imageData, imageShape);
                monitor.end();
                
                if (i == 0) {
                    printDetectionResults(results);
                }
                results.close();
            }
            
            monitor.printSummary();
            
        } finally {
            loader.close();
        }
    }
    
    /**
     * Test segmentation model
     */
    public static void testSegmentationModel(String modelPath, float[] inputData, long[] inputShape) 
            throws OrtException {
        logger.info("Testing segmentation model: {}", modelPath);
        
        OnnxModelLoader loader = new OnnxModelLoader(modelPath);
        PerformanceMonitor monitor = new PerformanceMonitor("Segmentation");
        
        try {
            loader.loadModel();
            monitor.start();
            
            OrtSession.Result results = runInferenceOnce(loader, inputData, inputShape);
            monitor.end();
            
            logger.info("Average inference time: {:.2f}ms", monitor.getAverageMs());
            
            printSegmentationResults(results);
            results.close();
            
        } finally {
            loader.close();
        }
    }
    
    /**
     * Test NLP model with text input
     */
    public static void testNLPModel(String modelPath, long[] inputIds, long[] attentionMask) 
            throws OrtException {
        logger.info("Testing NLP model: {}", modelPath);
        
        OnnxModelLoader loader = new OnnxModelLoader(modelPath);
        
        try {
            loader.loadModel();
            
            // Create input tensors
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            long[][] idArray = new long[][] {inputIds};
            long[][] maskArray = new long[][] {attentionMask};
            OnnxTensor idTensor = OnnxTensor.createTensor(env, idArray);
            OnnxTensor maskTensor = OnnxTensor.createTensor(env, maskArray);
            
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", idTensor);
            inputs.put("attention_mask", maskTensor);
            
            OrtSession.Result results = loader.runInference(inputs);
            
            logger.info("NLP inference completed");
            printNLPResults(results);
            
            idTensor.close();
            maskTensor.close();
            results.close();
            
        } finally {
            loader.close();
        }
    }
    
    /**
     * Run single inference iteration with error handling
     */
    private static OrtSession.Result runInferenceOnce(OnnxModelLoader loader, float[] inputData, long[] shape) 
            throws OrtException {
        OnnxTensor inputTensor = loader.createFloatTensor(inputData, shape);
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put(loader.getSession().getInputInfo().keySet().iterator().next(), inputTensor);
        
        OrtSession.Result results = loader.runInference(inputs);
        inputTensor.close();
        
        return results;
    }
    
    /**
     * Print detection results (bounding boxes, scores, etc.)
     */
    private static void printDetectionResults(OrtSession.Result results) throws OrtException {
        logger.info("Detection Results:");
        for (Map.Entry<String, OnnxValue> entry : results) {
            String outputName = entry.getKey();
            Object output = entry.getValue().getValue();
            
            if (output instanceof float[]) {
                float[] data = (float[]) output;
                logger.info("  {} - shape: [{}], sample: {}", 
                    outputName, data.length, 
                    Arrays.toString(Arrays.copyOfRange(data, 0, Math.min(10, data.length))));
            }
        }
    }
    
    /**
     * Print segmentation mask results
     */
    private static void printSegmentationResults(OrtSession.Result results) throws OrtException {
        logger.info("Segmentation Results:");
        for (Map.Entry<String, OnnxValue> entry : results) {
            String outputName = entry.getKey();
            Object output = entry.getValue().getValue();
            
            if (output instanceof long[]) {
                long[] masks = (long[]) output;
                logger.info("  {} - classes: {}", outputName, 
                    Arrays.toString(Arrays.copyOfRange(masks, 0, Math.min(10, masks.length))));
            }
        }
    }
    
    /**
     * Print NLP model results
     */
    private static void printNLPResults(OrtSession.Result results) throws OrtException {
        logger.info("NLP Results:");
        for (Map.Entry<String, OnnxValue> entry : results) {
            String outputName = entry.getKey();
            Object output = entry.getValue().getValue();
            
            if (output instanceof float[][]) {
                float[][] logits = (float[][]) output;
                logger.info("  {} - shape: [{}][{}]", outputName, logits.length, logits.length > 0 ? logits[0].length : 0);
            }
        }
    }
}
