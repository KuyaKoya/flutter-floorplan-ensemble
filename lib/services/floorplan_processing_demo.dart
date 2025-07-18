import 'dart:io';
import 'package:floorplan_detection_app/interfaces/detection.dart';
import 'package:floorplan_detection_app/interfaces/model_config.dart';
import 'package:floorplan_detection_app/services/pdf_processing_service.dart';
import 'package:floorplan_detection_app/services/tiled_room_detection_service.dart';
import 'package:image/image.dart' as img;

/// Demo service showing how to use the PDF import and tiling functionality
class FloorplanProcessingDemo {
  late TiledRoomDetectionService _detectionService;

  /// Initialize the demo with model configurations
  Future<void> initialize() async {
    print('=== Initializing Floorplan Processing Demo ===');

    // Configure multiple models for ensemble detection
    // You can add more models here with different weights
    final modelConfigs = [
      const ModelConfig(
        assetPath: 'assets/models/yolov5_rooms_v1.tflite',
        weight: 1.0,
      ),
      const ModelConfig(
        assetPath: 'assets/models/yolov5_rooms_v2.tflite',
        weight: 0.8,
      ),
    ];

    _detectionService = TiledRoomDetectionService(modelConfigs: modelConfigs);

    // Load the models
    final success = await _detectionService.loadModels();
    if (success) {
      print('✓ Models loaded successfully');
    } else {
      print('✗ Failed to load models');
      throw Exception('Model loading failed');
    }
  }

  /// Complete workflow: PDF selection -> processing -> room detection
  Future<ProcessingResult> processFloorplanWorkflow() async {
    print('=== Starting complete floorplan processing workflow ===');

    try {
      // Step 1: Let user pick a PDF file
      print('Step 1: Selecting PDF file...');
      final pdfFile = await PdfProcessingService.pickPdfFile();
      if (pdfFile == null) {
        throw Exception('No PDF file selected');
      }
      print('✓ Selected PDF: ${pdfFile.path}');

      // Step 2: Process the PDF with room detection
      final result = await processPdfFile(pdfFile);
      print('✓ Processing completed successfully');

      return result;
    } catch (e) {
      print('✗ Workflow failed: $e');
      rethrow;
    }
  }

  /// Process a specific PDF file
  Future<ProcessingResult> processPdfFile(File pdfFile,
      {double dpi = 150.0}) async {
    print('=== Processing PDF file: ${pdfFile.path} ===');

    final stopwatch = Stopwatch()..start();

    try {
      // Convert PDF to image
      print('Converting PDF to image...');
      final image = await PdfProcessingService.pdfToImage(pdfFile, dpi: dpi);
      if (image == null) {
        throw Exception('Failed to convert PDF to image');
      }

      final conversionTime = stopwatch.elapsedMilliseconds;
      print(
          '✓ PDF converted in ${conversionTime}ms - Image: ${image.width}x${image.height}');

      // Process with tiled room detection
      print('Running tiled room detection...');
      final detectionStart = stopwatch.elapsedMilliseconds;

      final detections = await _detectionService.processLargeImage(image);

      final detectionTime = stopwatch.elapsedMilliseconds - detectionStart;
      final totalTime = stopwatch.elapsedMilliseconds;

      print('✓ Detection completed in ${detectionTime}ms');
      print('✓ Total processing time: ${totalTime}ms');

      // Create result summary
      final result = ProcessingResult(
        sourceFile: pdfFile,
        processedImage: image,
        detections: detections,
        processingTimeMs: totalTime,
        conversionTimeMs: conversionTime,
        detectionTimeMs: detectionTime,
        dpi: dpi,
      );

      _logProcessingResult(result);
      return result;
    } catch (e) {
      print('✗ PDF processing failed: $e');
      rethrow;
    }
  }

  /// Process a regular image file (non-PDF) for comparison
  Future<ProcessingResult> processImageFile(File imageFile) async {
    print('=== Processing image file: ${imageFile.path} ===');

    final stopwatch = Stopwatch()..start();

    try {
      // Load image
      final bytes = await imageFile.readAsBytes();
      final image = img.decodeImage(bytes);
      if (image == null) {
        throw Exception('Failed to decode image file');
      }

      print('✓ Image loaded: ${image.width}x${image.height}');

      // Process with tiled room detection
      final detections = await _detectionService.processLargeImage(image);

      final totalTime = stopwatch.elapsedMilliseconds;

      final result = ProcessingResult(
        sourceFile: imageFile,
        processedImage: image,
        detections: detections,
        processingTimeMs: totalTime,
        conversionTimeMs: 0, // No conversion needed for images
        detectionTimeMs: totalTime,
        dpi: 0, // Not applicable for direct images
      );

      _logProcessingResult(result);
      return result;
    } catch (e) {
      print('✗ Image processing failed: $e');
      rethrow;
    }
  }

  /// Log detailed processing results
  void _logProcessingResult(ProcessingResult result) {
    print('=== Processing Result Summary ===');
    print('Source: ${result.sourceFile.path}');
    print(
        'Image size: ${result.processedImage.width}x${result.processedImage.height}');
    print('Detections found: ${result.detections.length}');
    print('Total time: ${result.processingTimeMs}ms');
    if (result.conversionTimeMs > 0) {
      print('PDF conversion: ${result.conversionTimeMs}ms');
      print('Detection only: ${result.detectionTimeMs}ms');
      print('DPI: ${result.dpi}');
    }

    if (result.detections.isNotEmpty) {
      print('Detection details:');
      for (int i = 0; i < result.detections.length; i++) {
        final d = result.detections[i];
        print('  Room ${i + 1}: confidence=${d.confidence.toStringAsFixed(3)}, '
            'area=${(d.width * d.height).toStringAsFixed(0)}px²');
      }
    }

    print('=== End Result Summary ===');
  }

  /// Demonstrate different processing scenarios
  Future<void> runDemoScenarios() async {
    print('=== Running Demo Scenarios ===');

    // Scenario 1: High DPI processing
    print('\n--- Scenario 1: High DPI PDF Processing ---');
    try {
      final pdfFile = await PdfProcessingService.pickPdfFile();
      if (pdfFile != null) {
        await processPdfFile(pdfFile, dpi: 200.0);
      }
    } catch (e) {
      print('Scenario 1 failed: $e');
    }

    // Scenario 2: Standard DPI processing
    print('\n--- Scenario 2: Standard DPI PDF Processing ---');
    try {
      final pdfFile = await PdfProcessingService.pickPdfFile();
      if (pdfFile != null) {
        await processPdfFile(pdfFile, dpi: 150.0);
      }
    } catch (e) {
      print('Scenario 2 failed: $e');
    }

    print('=== Demo Scenarios Completed ===');
  }

  /// Cleanup resources
  void dispose() {
    _detectionService.dispose();
  }
}

/// Result of processing a floorplan document
class ProcessingResult {
  final File sourceFile;
  final img.Image processedImage;
  final List<Detection> detections;
  final int processingTimeMs;
  final int conversionTimeMs;
  final int detectionTimeMs;
  final double dpi;

  const ProcessingResult({
    required this.sourceFile,
    required this.processedImage,
    required this.detections,
    required this.processingTimeMs,
    required this.conversionTimeMs,
    required this.detectionTimeMs,
    required this.dpi,
  });

  /// Get processing efficiency metrics
  Map<String, dynamic> getMetrics() {
    final imagePixels = processedImage.width * processedImage.height;
    final pixelsPerMs = imagePixels / processingTimeMs;
    final detectionsPerMegapixel = detections.length / (imagePixels / 1000000);

    return {
      'image_size': '${processedImage.width}x${processedImage.height}',
      'total_pixels': imagePixels,
      'processing_time_ms': processingTimeMs,
      'pixels_per_ms': pixelsPerMs.toStringAsFixed(2),
      'detections_count': detections.length,
      'detections_per_megapixel': detectionsPerMegapixel.toStringAsFixed(2),
      'average_confidence': detections.isEmpty
          ? 0.0
          : (detections.map((d) => d.confidence).reduce((a, b) => a + b) /
              detections.length),
    };
  }
}
