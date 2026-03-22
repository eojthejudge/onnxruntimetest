package com.onnx.inference.util;

/**
 * Performance monitoring utility for inference benchmarking
 */
public class PerformanceMonitor {
    private long startTime;
    private long endTime;
    private final String name;
    private int iterations;
    private long totalTime;
    
    public PerformanceMonitor(String name) {
        this.name = name;
        this.iterations = 0;
        this.totalTime = 0;
    }
    
    /**
     * Start timing
     */
    public void start() {
        this.startTime = System.nanoTime();
    }
    
    /**
     * End timing and record
     */
    public void end() {
        this.endTime = System.nanoTime();
        long elapsed = endTime - startTime;
        totalTime += elapsed;
        iterations++;
    }
    
    /**
     * Get latest iteration time in milliseconds
     */
    public double getLastIterationMs() {
        return (endTime - startTime) / 1_000_000.0;
    }
    
    /**
     * Get average time in milliseconds
     */
    public double getAverageMs() {
        if (iterations == 0) return 0;
        return totalTime / iterations / 1_000_000.0;
    }
    
    /**
     * Get min time in milliseconds
     */
    public double getMinMs() {
        return (endTime - startTime) / 1_000_000.0;
    }
    
    /**
     * Print summary statistics
     */
    public void printSummary() {
        System.out.println("\n--- Performance Summary: " + name + " ---");
        System.out.printf("Iterations: %d%n", iterations);
        System.out.printf("Average time: %.3f ms%n", getAverageMs());
        System.out.printf("Total time: %.3f ms%n", totalTime / 1_000_000.0);
        System.out.printf("Throughput: %.2f inferences/sec%n", 1000.0 / getAverageMs());
    }
    
    /**
     * Reset counters
     */
    public void reset() {
        this.iterations = 0;
        this.totalTime = 0;
    }
}
