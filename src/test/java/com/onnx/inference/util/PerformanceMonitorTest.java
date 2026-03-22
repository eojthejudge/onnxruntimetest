package com.onnx.inference.util;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for PerformanceMonitor
 */
public class PerformanceMonitorTest {
    
    @Test
    public void testPerformanceMonitorBasics() throws InterruptedException {
        PerformanceMonitor monitor = new PerformanceMonitor("Test Monitor");
        
        // Run some iterations
        for (int i = 0; i < 5; i++) {
            monitor.start();
            Thread.sleep(10); // Simulate some work
            monitor.end();
        }
        
        // Verify calculations
        double avgTime = monitor.getAverageMs();
        assertTrue(avgTime >= 10, "Average time should be at least 10ms");
        
        double lastTime = monitor.getLastIterationMs();
        assertTrue(lastTime >= 10, "Last iteration time should be at least 10ms");
    }
    
    @Test
    public void testPerformanceMonitorReset() {
        PerformanceMonitor monitor = new PerformanceMonitor("Test");
        
        monitor.start();
        monitor.end();
        
        monitor.reset();
        // After reset, average should be 0
        assertEquals(0, monitor.getAverageMs());
    }
    
    @Test
    public void testEmptyMonitor() {
        PerformanceMonitor monitor = new PerformanceMonitor("Empty");
        assertEquals(0, monitor.getAverageMs(), "Empty monitor should have 0 average");
    }
}
