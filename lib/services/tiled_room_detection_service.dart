import 'dart:io';
import 'package:floorplan_detection_app/interfaces/detection.dart';
import 'package:floorplan_detection_app/interfaces/model_config.dart';
import 'package:floorplan_detection_app/services/room_detection_service.dart';
import 'package:floorplan_detection_app/services/pdf_processing_service.dart';
import 'package:image/image.dart' as img;

/// Enhanced room detection service with PDF import and tiling support
class TiledRoomDetectionService {
  final RoomDetectionService _baseDetectionService;

  TiledRoomDetectionService({required List<ModelConfig> modelConfigs})
      : _baseDetectionService =
            RoomDetectionService(modelConfigs: modelConfigs);

  /// Initialize the underlying detection models
  Future<bool> loadModels() async {
    return await _baseDetectionService.loadModels();
  }

  /// Check if models are loaded
  bool get isModelLoaded => _baseDetectionService.isModelLoaded;

  /// Process a PDF file and detect rooms using tiling approach
  Future<List<Detection>> processPDF(File pdfFile, {double dpi = 150.0}) async {
    print('=== Starting PDF processing ===');

    // Step 1: Convert PDF to image
    final image = await PdfProcessingService.pdfToImage(pdfFile, dpi: dpi);
    if (image == null) {
      throw Exception('Failed to convert PDF to image');
    }

    print('PDF converted to image: ${image.width}x${image.height}');

    // Step 2: Process the image using tiling
    return await processLargeImage(image);
  }

  /// Process a large image using tiling approach for room detection
  Future<List<Detection>> processLargeImage(img.Image image) async {
    print('=== Starting tiled room detection ===');
    print('Input image size: ${image.width}x${image.height}');

    // Step 1: Create tiles
    final tiles = PdfProcessingService.createImageTiles(image);
    print('Created ${tiles.length} tiles for processing');

    // Step 2: Process each tile
    List<Detection> allDetections = [];
    int processedTiles = 0;

    for (final tile in tiles) {
      print('Processing tile ${tile.index + 1}/${tiles.length}: $tile');

      try {
        // Run detection on the tile
        final tileDetections =
            await _baseDetectionService.runInference(tile.image);
        print(
            'Tile ${tile.index} produced ${tileDetections.length} raw detections');

        // Calculate processing bounds to filter out overlap regions
        final processingBounds = PdfProcessingService.calculateProcessingBounds(
            tile, image.width, image.height);
        print('Processing bounds for tile ${tile.index}: $processingBounds');

        // Transform detections to global coordinates and filter by processing bounds
        final globalDetections = _transformTileDetectionsToGlobal(
            tileDetections, tile, processingBounds);

        print(
            'Tile ${tile.index} contributed ${globalDetections.length} global detections');
        allDetections.addAll(globalDetections);
        processedTiles++;
      } catch (e) {
        print('Error processing tile ${tile.index}: $e');
        // Continue with other tiles even if one fails
      }
    }

    print('Processed $processedTiles/${tiles.length} tiles successfully');
    print(
        'Total raw detections before global processing: ${allDetections.length}');

    // Step 3: Apply global post-processing
    final finalDetections = _applyGlobalPostProcessing(allDetections);

    print(
        'Final detections after global post-processing: ${finalDetections.length}');
    print('=== Tiled room detection completed ===');

    return finalDetections;
  }

  /// Transform tile-local detections to global image coordinates
  List<Detection> _transformTileDetectionsToGlobal(
    List<Detection> tileDetections,
    ImageTile tile,
    ProcessingBounds processingBounds,
  ) {
    List<Detection> globalDetections = [];

    for (final detection in tileDetections) {
      // Check if detection center is within the processing bounds
      final detectionCenterX = detection.left + detection.width / 2;
      final detectionCenterY = detection.top + detection.height / 2;

      final isWithinBounds = detectionCenterX >= processingBounds.left &&
          detectionCenterX < processingBounds.left + processingBounds.width &&
          detectionCenterY >= processingBounds.top &&
          detectionCenterY < processingBounds.top + processingBounds.height;

      if (!isWithinBounds) {
        // Skip detections outside processing bounds to avoid double-counting
        continue;
      }

      // Transform coordinates to global image space
      final globalLeft = detection.left + tile.offsetX;
      final globalTop = detection.top + tile.offsetY;

      // Ensure detection stays within original image bounds
      final clampedLeft = globalLeft.clamp(0.0, double.infinity);
      final clampedTop = globalTop.clamp(0.0, double.infinity);

      final globalDetection = Detection(
        left: clampedLeft,
        top: clampedTop,
        width: detection.width,
        height: detection.height,
        confidence: detection.confidence,
        classId: detection.classId,
        label: detection.label,
      );

      globalDetections.add(globalDetection);

      print('Transformed detection from tile ${tile.index}: '
          'local(${detection.left.toStringAsFixed(1)}, ${detection.top.toStringAsFixed(1)}) -> '
          'global(${globalLeft.toStringAsFixed(1)}, ${globalTop.toStringAsFixed(1)})');
    }

    return globalDetections;
  }

  /// Apply global post-processing to merge and clean up detections
  List<Detection> _applyGlobalPostProcessing(List<Detection> allDetections) {
    print(
        'Applying global post-processing to ${allDetections.length} detections');

    // Step 1: Apply weighted box averaging for overlapping detections
    print('Step 1: Applying weighted box averaging');
    final averagedDetections =
        _baseDetectionService.averageOverlappingBoxes(allDetections);
    print('After averaging: ${averagedDetections.length} detections');

    // Step 2: Apply Non-Maximum Suppression
    print('Step 2: Applying Non-Maximum Suppression');
    final finalDetections = _baseDetectionService.applyNMS(averagedDetections);
    print('After NMS: ${finalDetections.length} detections');

    // Step 3: Log final detection statistics
    _logDetectionStatistics(finalDetections);

    return finalDetections;
  }

  /// Log detection statistics for debugging
  void _logDetectionStatistics(List<Detection> detections) {
    if (detections.isEmpty) {
      print('No detections found');
      return;
    }

    print('=== Final Detection Statistics ===');

    // Calculate confidence statistics
    final confidences = detections.map((d) => d.confidence).toList();
    confidences.sort();

    final minConfidence = confidences.first;
    final maxConfidence = confidences.last;
    final avgConfidence =
        confidences.reduce((a, b) => a + b) / confidences.length;
    final medianConfidence = confidences[confidences.length ~/ 2];

    print('Confidence stats: min=${minConfidence.toStringAsFixed(3)}, '
        'max=${maxConfidence.toStringAsFixed(3)}, '
        'avg=${avgConfidence.toStringAsFixed(3)}, '
        'median=${medianConfidence.toStringAsFixed(3)}');

    // Calculate size statistics
    final areas = detections.map((d) => d.width * d.height).toList();
    areas.sort();

    final minArea = areas.first;
    final maxArea = areas.last;
    final avgArea = areas.reduce((a, b) => a + b) / areas.length;

    print('Area stats: min=${minArea.toStringAsFixed(0)}, '
        'max=${maxArea.toStringAsFixed(0)}, '
        'avg=${avgArea.toStringAsFixed(0)}');

    // List individual detections
    print('Individual detections:');
    for (int i = 0; i < detections.length; i++) {
      final d = detections[i];
      print('  $i: ${d.label} conf=${d.confidence.toStringAsFixed(3)} '
          'box=[${d.left.toStringAsFixed(1)}, ${d.top.toStringAsFixed(1)}, '
          '${d.width.toStringAsFixed(1)}, ${d.height.toStringAsFixed(1)}] '
          'area=${(d.width * d.height).toStringAsFixed(0)}');
    }

    print('=== End Statistics ===');
  }

  /// Dispose resources
  void dispose() {
    _baseDetectionService.dispose();
  }
}
