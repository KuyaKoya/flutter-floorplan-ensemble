import 'package:floorplan_detection_app/interfaces/model_config.dart';
import 'package:floorplan_detection_app/services/floorplan_processing_demo.dart';
import 'package:floorplan_detection_app/services/pdf_processing_service.dart';

/// Example usage of the enhanced room detection pipeline
class RoomDetectionExample {
  /// Example 1: Basic PDF processing
  static Future<void> basicPdfProcessing() async {
    print('=== Example 1: Basic PDF Processing ===');

    // Initialize the demo service
    final demo = FloorplanProcessingDemo();
    await demo.initialize();

    try {
      // Let user pick a PDF file
      final pdfFile = await PdfProcessingService.pickPdfFile();
      if (pdfFile == null) {
        print('No file selected');
        return;
      }

      // Process the PDF
      final result = await demo.processPdfFile(pdfFile);

      print('Processing completed!');
      print('Found ${result.detections.length} rooms');
      print('Total time: ${result.processingTimeMs}ms');
    } finally {
      demo.dispose();
    }
  }

  /// Example 2: High-resolution PDF processing
  static Future<void> highResPdfProcessing() async {
    print('=== Example 2: High-Resolution PDF Processing ===');

    final demo = FloorplanProcessingDemo();
    await demo.initialize();

    try {
      final pdfFile = await PdfProcessingService.pickPdfFile();
      if (pdfFile == null) return;

      // Process at high DPI for better detail
      final result = await demo.processPdfFile(pdfFile, dpi: 200.0);

      print('High-res processing completed!');
      print(
          'Image size: ${result.processedImage.width}x${result.processedImage.height}');
      print('Detections: ${result.detections.length}');

      // Show metrics
      final metrics = result.getMetrics();
      print('Processing efficiency: ${metrics['pixels_per_ms']} pixels/ms');
      print(
          'Detection density: ${metrics['detections_per_megapixel']} per megapixel');
    } finally {
      demo.dispose();
    }
  }

  /// Example 3: Multi-model ensemble processing
  static Future<void> ensembleProcessing() async {
    print('=== Example 3: Multi-Model Ensemble Processing ===');

    // Configure multiple models with different weights
    final modelConfigs = [
      const ModelConfig(
        assetPath: 'assets/models/yolov5_primary.tflite',
        weight: 1.0, // Primary model
      ),
      const ModelConfig(
        assetPath: 'assets/models/yolov5_secondary.tflite',
        weight: 0.7, // Secondary model with lower weight
      ),
      const ModelConfig(
        assetPath: 'assets/models/yolov5_specialized.tflite',
        weight: 0.5, // Specialized model for specific room types
      ),
    ];

    // Note: This example shows the configuration but won't run without actual model files
    print('Configured ${modelConfigs.length} models for ensemble detection');
    print('Model weights: ${modelConfigs.map((m) => m.weight).join(', ')}');
  }

  /// Example 4: Performance comparison
  static Future<void> performanceComparison() async {
    print('=== Example 4: Performance Comparison ===');

    final demo = FloorplanProcessingDemo();
    await demo.initialize();

    try {
      final pdfFile = await PdfProcessingService.pickPdfFile();
      if (pdfFile == null) return;

      // Test different DPI settings
      final dpiSettings = [100.0, 150.0, 200.0];

      for (final dpi in dpiSettings) {
        print('\n--- Testing DPI: $dpi ---');

        final result = await demo.processPdfFile(pdfFile, dpi: dpi);
        final metrics = result.getMetrics();

        print('Image size: ${metrics['image_size']}');
        print('Processing time: ${metrics['processing_time_ms']}ms');
        print('Efficiency: ${metrics['pixels_per_ms']} pixels/ms');
        print('Detections: ${metrics['detections_count']}');
      }
    } finally {
      demo.dispose();
    }
  }
}

/// Usage instructions and tips
class UsageInstructions {
  static void printInstructions() {
    print('''
=== Room Detection Pipeline Usage Instructions ===

1. SETUP:
   - Place your YOLO TensorFlow Lite models in assets/models/
   - Update model paths in ModelConfig objects
   - Ensure models are trained for room detection with class ID 0

2. PDF PROCESSING:
   - Use PdfProcessingService.pickPdfFile() to select files
   - Adjust DPI parameter based on detail requirements:
     * 100 DPI: Fast processing, lower detail
     * 150 DPI: Balanced (recommended)
     * 200+ DPI: High detail, slower processing

3. TILING SYSTEM:
   - Automatically splits large images into 640×640 tiles
   - 64px overlap prevents edge detection issues
   - Coordinate transformation maps results back to full image

4. OPTIMIZATION TIPS:
   - Start with lower DPI for testing
   - Use ensemble models for better accuracy
   - Monitor processing times for large documents
   - Adjust confidence thresholds in RoomDetectionService

5. DEBUGGING:
   - Enable verbose logging in services
   - Check tile creation and processing bounds
   - Verify model loading and inference results
   - Use FloorplanProcessingDemo for complete workflows

=== End Instructions ===
    ''');
  }
}
