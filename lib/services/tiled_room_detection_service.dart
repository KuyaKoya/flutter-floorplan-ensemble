import 'dart:io';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:floorplan_detection_app/interfaces/detection.dart';
import 'package:floorplan_detection_app/services/base_room_detection_service.dart';
import 'package:floorplan_detection_app/services/pdf_processing_service.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

/// Tiled room detection service for processing large images
/// Divides images into overlapping tiles for better detection of large floor plans
class TiledRoomDetectionService extends BaseRoomDetectionService {
  static const int tileSize = 640;
  static const int overlap = 64;
  static const double confidenceThreshold = 0.1;
  static const double iouGroupingThreshold = 0.5;
  static const double nmsThreshold = 0.45;

  static const List<String> roomClassNames = ['room'];

  final List<Interpreter> _interpreters = [];
  bool _isModelLoaded = false;

  TiledRoomDetectionService({
    required super.modelConfigs,
    super.onLog,
  });

  @override
  Future<bool> loadModels() async {
    try {
      logMessage(
          'Loading ${modelConfigs.length} model(s) for tiled processing...');

      for (int i = 0; i < modelConfigs.length; i++) {
        final config = modelConfigs[i];
        logMessage('Loading model ${i + 1}: ${config.assetPath}');

        final interpreter = await Interpreter.fromAsset(config.assetPath);
        _interpreters.add(interpreter);

        logMessage('✓ Model ${i + 1} loaded successfully');
      }

      _isModelLoaded = true;
      logMessage(
          '✓ All ${_interpreters.length} models loaded for tiled processing');
      return true;
    } catch (e) {
      logMessage('✗ Failed to load models: $e');
      _isModelLoaded = false;
      return false;
    }
  }

  @override
  bool get isModelLoaded => _isModelLoaded;

  @override
  Future<List<Detection>> processPDF(File pdfFile, {double dpi = 150.0}) async {
    try {
      logMessage('=== Starting PDF Processing (Tiled) ===');
      logMessage('Input PDF: ${pdfFile.path}');

      // Step 1: Convert PDF to image
      logMessage('Converting PDF to image (DPI: $dpi)...');
      final image = await PdfProcessingService.pdfToImage(pdfFile, dpi: dpi);
      if (image == null) {
        throw Exception('Failed to convert PDF to image');
      }

      logMessage(
          '✓ PDF converted to image: ${image.width}x${image.height} pixels');

      // Step 2: Process the image using tiling
      return await processLargeImage(image);
    } catch (e) {
      logMessage('✗ Error processing PDF: $e');
      rethrow;
    }
  }

  @override
  Future<List<Detection>> processImageFile(File imageFile) async {
    try {
      logMessage('=== Starting Image Processing (Tiled) ===');
      logMessage('Input image: ${imageFile.path}');

      // Step 1: Load and decode image
      logMessage('Loading and decoding image...');
      final imageBytes = await imageFile.readAsBytes();
      logMessage('Image file size: ${imageBytes.length} bytes');

      final image = await compute(_decodeImage, imageBytes);
      if (image == null) {
        throw Exception('Failed to decode image');
      }

      logMessage('✓ Image decoded: ${image.width}x${image.height} pixels');

      // Step 2: Process the image using tiling
      return await processLargeImage(image);
    } catch (e) {
      logMessage('✗ Error processing image: $e');
      rethrow;
    }
  }

  /// Process a large image using tiling approach for room detection
  Future<List<Detection>> processLargeImage(img.Image image) async {
    try {
      logMessage('=== Starting Tiled Room Detection ===');
      logMessage('Input image size: ${image.width}x${image.height} pixels');

      if (!_isModelLoaded || _interpreters.isEmpty) {
        throw Exception('Models not loaded');
      }

      // Step 1: Create tiles
      logMessage(
          'Creating tiles (${tileSize}x$tileSize with ${overlap}px overlap)...');
      final tiles = await compute(_createImageTiles, image);
      logMessage('✓ Created ${tiles.length} tiles for processing');

      if (tiles.isEmpty) {
        logMessage('⚠️ No tiles created - image may be too small');
        return [];
      }

      // Step 2: Process each tile asynchronously
      logMessage('Processing tiles with ${_interpreters.length} model(s)...');
      List<Detection> allDetections = [];

      for (int tileIndex = 0; tileIndex < tiles.length; tileIndex++) {
        final tile = tiles[tileIndex];
        final progress =
            ((tileIndex + 1) / tiles.length * 100).toStringAsFixed(1);

        logMessage(
            'Processing tile ${tileIndex + 1}/${tiles.length} ($progress%) at offset (${tile.offsetX}, ${tile.offsetY})');

        try {
          // Process tile with all models
          final tileDetections = await _processTileAsync(tile, image);

          if (tileDetections.isNotEmpty) {
            logMessage(
                '  ✓ Found ${tileDetections.length} detections in tile ${tileIndex + 1}');
            allDetections.addAll(tileDetections);
          } else {
            logMessage('  - No detections in tile ${tileIndex + 1}');
          }
        } catch (e) {
          logMessage('  ✗ Error processing tile ${tileIndex + 1}: $e');
          // Continue with other tiles
        }
      }

      logMessage(
          '✓ Tile processing completed: ${allDetections.length} total detections');

      // Step 3: Post-process all detections
      return await _postProcessDetections(allDetections);
    } catch (e) {
      logMessage('✗ Error in tiled detection: $e');
      rethrow;
    }
  }

  @override
  void dispose() {
    logMessage('Disposing tiled detection service resources...');
    for (final interpreter in _interpreters) {
      interpreter.close();
    }
    _interpreters.clear();
    _isModelLoaded = false;
  }

  // Static methods for isolate processing

  /// Decode image in isolate
  static img.Image? _decodeImage(Uint8List imageBytes) {
    return img.decodeImage(imageBytes);
  }

  /// Create image tiles in isolate
  static List<TileData> _createImageTiles(img.Image image) {
    List<TileData> tiles = [];

    for (int y = 0; y < image.height; y += (tileSize - overlap)) {
      for (int x = 0; x < image.width; x += (tileSize - overlap)) {
        final endX = (x + tileSize).clamp(0, image.width);
        final endY = (y + tileSize).clamp(0, image.height);

        if (endX - x < tileSize ~/ 2 || endY - y < tileSize ~/ 2) {
          continue; // Skip very small tiles
        }

        // Extract and resize tile
        final tile =
            img.copyCrop(image, x: x, y: y, width: endX - x, height: endY - y);
        final resizedTile =
            img.copyResize(tile, width: tileSize, height: tileSize);

        tiles.add(TileData(
          image: resizedTile,
          offsetX: x,
          offsetY: y,
          originalWidth: endX - x,
          originalHeight: endY - y,
        ));
      }
    }

    return tiles;
  }

  /// Process a single tile with all models
  Future<List<Detection>> _processTileAsync(
      TileData tile, img.Image originalImage) async {
    List<Detection> allDetections = [];

    try {
      // Preprocess the tile
      final preprocessedData = await compute(_preprocessImage, tile.image);

      // Run inference on each model
      for (int modelIndex = 0;
          modelIndex < _interpreters.length;
          modelIndex++) {
        final interpreter = _interpreters[modelIndex];
        final weight = modelConfigs[modelIndex].weight;

        try {
          // Create output buffers
          final outputs = _createOutputBuffers(interpreter);

          // Run inference
          final inputData = preprocessedData['float32List'] as Float32List;
          interpreter.run(inputData, outputs[0]);

          // Post-process detections for this model
          final modelDetections = await compute(_postprocessTileDetections, {
            'output': outputs[0],
            'outputShape': interpreter.getOutputTensor(0).shape,
            'tileOffsetX': tile.offsetX.toDouble(),
            'tileOffsetY': tile.offsetY.toDouble(),
            'originalWidth': originalImage.width.toDouble(),
            'originalHeight': originalImage.height.toDouble(),
            'weight': weight,
            'confidenceThreshold': confidenceThreshold,
          });

          allDetections.addAll(modelDetections);
        } catch (e) {
          logMessage('    ✗ Model ${modelIndex + 1} failed on tile: $e');
          // Continue with other models
        }
      }
    } catch (e) {
      logMessage('    ✗ Tile preprocessing failed: $e');
    }

    return allDetections;
  }

  /// Post-process all detections with NMS and overlap handling
  Future<List<Detection>> _postProcessDetections(
      List<Detection> allDetections) async {
    try {
      logMessage('Post-processing detections...');
      logMessage('Total raw detections: ${allDetections.length}');

      if (allDetections.isEmpty) {
        logMessage('⚠️ No detections found from any tile');
        return [];
      }

      // Step 1: Average overlapping detections
      logMessage('Averaging overlapping detections...');
      final averagedDetections = await compute(_averageOverlappingBoxes, {
        'detections': allDetections,
        'iouThreshold': iouGroupingThreshold,
      });
      logMessage('Detections after averaging: ${averagedDetections.length}');

      // Step 2: Apply NMS
      logMessage('Applying non-maximum suppression...');
      final finalDetections = await compute(_applyNMS, {
        'detections': averagedDetections,
        'nmsThreshold': nmsThreshold,
      });

      logMessage(
          '✓ Final processing completed: ${finalDetections.length} detections');

      // Log final results
      if (finalDetections.isNotEmpty) {
        logMessage('=== Final Detection Results ===');
        for (int i = 0; i < finalDetections.length; i++) {
          final detection = finalDetections[i];
          logMessage(
              'Room ${i + 1}: ${detection.label} (${(detection.confidence * 100).toStringAsFixed(1)}% confidence)');
        }
      }

      return finalDetections;
    } catch (e) {
      logMessage('✗ Error in post-processing: $e');
      return allDetections; // Return unprocessed detections as fallback
    }
  }

  /// Preprocess image for model input in isolate
  static Map<String, dynamic> _preprocessImage(img.Image image) {
    final inputBytes = Float32List(1 * tileSize * tileSize * 3);
    int pixelIndex = 0;

    for (int y = 0; y < tileSize; y++) {
      for (int x = 0; x < tileSize; x++) {
        final pixel = image.getPixel(x, y);
        inputBytes[pixelIndex++] = pixel.r / 255.0;
        inputBytes[pixelIndex++] = pixel.g / 255.0;
        inputBytes[pixelIndex++] = pixel.b / 255.0;
      }
    }

    return {
      'float32List': inputBytes,
    };
  }

  /// Create output buffers for TensorFlow Lite models
  List<dynamic> _createOutputBuffers(Interpreter interpreter) {
    List<dynamic> outputs = [];
    final outputTensor = interpreter.getOutputTensor(0);
    final shape = outputTensor.shape;

    if (shape.length == 3) {
      outputs.add(List.generate(shape[0],
          (b) => List.generate(shape[1], (d) => List.filled(shape[2], 0.0))));
    } else if (shape.length == 2) {
      outputs.add(List.generate(shape[0], (f) => List.filled(shape[1], 0.0)));
    } else {
      throw Exception('Unsupported output shape: $shape');
    }

    return outputs;
  }

  /// Post-process detections from a single tile
  static List<Detection> _postprocessTileDetections(
      Map<String, dynamic> params) {
    final output = params['output'];
    final outputShape = params['outputShape'] as List<int>;
    final tileOffsetX = params['tileOffsetX'] as double;
    final tileOffsetY = params['tileOffsetY'] as double;
    final originalWidth = params['originalWidth'] as double;
    final originalHeight = params['originalHeight'] as double;
    final weight = params['weight'] as double;
    final confThreshold = params['confidenceThreshold'] as double;

    List<Detection> detections = [];

    try {
      if (outputShape.length == 3) {
        final dim1 = outputShape[1];
        final dim2 = outputShape[2];

        if (dim1 > dim2) {
          // YOLOv5 format
          _processYOLOv5FormatTiled(
              output,
              dim1,
              dim2,
              tileOffsetX,
              tileOffsetY,
              originalWidth,
              originalHeight,
              weight,
              detections,
              confThreshold);
        } else {
          // YOLOv8 format
          _processYOLOv8FormatTiled(
              output,
              dim1,
              dim2,
              tileOffsetX,
              tileOffsetY,
              originalWidth,
              originalHeight,
              weight,
              detections,
              confThreshold);
        }
      }
    } catch (e) {
      print('Error in tile postprocessing: $e');
    }

    return detections;
  }

  /// Average overlapping boxes in isolate
  static List<Detection> _averageOverlappingBoxes(Map<String, dynamic> params) {
    final detections = params['detections'] as List<Detection>;
    final iouThreshold = params['iouThreshold'] as double;

    if (detections.isEmpty) return detections;

    List<Detection> averaged = [];
    List<bool> used = List.filled(detections.length, false);

    for (int i = 0; i < detections.length; i++) {
      if (used[i]) continue;

      List<Detection> group = [detections[i]];
      used[i] = true;

      for (int j = i + 1; j < detections.length; j++) {
        if (used[j]) continue;

        final iou = _calculateIoU(detections[i], detections[j]);
        if (iou > iouThreshold) {
          group.add(detections[j]);
          used[j] = true;
        }
      }

      if (group.length == 1) {
        averaged.add(group[0]);
      } else {
        final avgLeft =
            group.map((d) => d.left).reduce((a, b) => a + b) / group.length;
        final avgTop =
            group.map((d) => d.top).reduce((a, b) => a + b) / group.length;
        final avgWidth =
            group.map((d) => d.width).reduce((a, b) => a + b) / group.length;
        final avgHeight =
            group.map((d) => d.height).reduce((a, b) => a + b) / group.length;
        final avgConfidence =
            group.map((d) => d.confidence).reduce((a, b) => a + b) /
                group.length;

        averaged.add(Detection(
          left: avgLeft,
          top: avgTop,
          width: avgWidth,
          height: avgHeight,
          confidence: avgConfidence,
          classId: group[0].classId,
          label: group[0].label,
        ));
      }
    }

    return averaged;
  }

  /// Apply NMS in isolate
  static List<Detection> _applyNMS(Map<String, dynamic> params) {
    final detections = params['detections'] as List<Detection>;
    final nmsThreshold = params['nmsThreshold'] as double;

    if (detections.isEmpty) return detections;

    // Sort by confidence
    final sortedDetections = List<Detection>.from(detections);
    sortedDetections.sort((a, b) => b.confidence.compareTo(a.confidence));

    List<Detection> kept = [];
    List<bool> suppressed = List.filled(sortedDetections.length, false);

    for (int i = 0; i < sortedDetections.length; i++) {
      if (suppressed[i]) continue;

      kept.add(sortedDetections[i]);

      for (int j = i + 1; j < sortedDetections.length; j++) {
        if (suppressed[j]) continue;

        final iou = _calculateIoU(sortedDetections[i], sortedDetections[j]);
        if (iou > nmsThreshold) {
          suppressed[j] = true;
        }
      }
    }

    return kept;
  }

  /// Calculate IoU between two detections
  static double _calculateIoU(Detection a, Detection b) {
    final left = max(a.left, b.left);
    final top = max(a.top, b.top);
    final right = min(a.left + a.width, b.left + b.width);
    final bottom = min(a.top + a.height, b.top + b.height);

    if (right <= left || bottom <= top) return 0.0;

    final intersection = (right - left) * (bottom - top);
    final areaA = a.width * a.height;
    final areaB = b.width * b.height;
    final union = areaA + areaB - intersection;

    return intersection / union;
  }

  /// Process YOLOv5 format detections for tiles
  static void _processYOLOv5FormatTiled(
      dynamic output,
      int numDetections,
      int numFeatures,
      double tileOffsetX,
      double tileOffsetY,
      double originalWidth,
      double originalHeight,
      double weight,
      List<Detection> detections,
      double confThreshold) {
    for (int i = 0; i < numDetections; i++) {
      try {
        final detectionData = output[0]?[i];
        if (detectionData == null || detectionData.length < 5) continue;

        final objectness = detectionData[4]?.toDouble() ?? 0.0;
        double finalConfidence = objectness * weight;

        if (numFeatures > 5 && detectionData.length > 5) {
          final classScore = detectionData[5]?.toDouble() ?? 1.0;
          finalConfidence = objectness * classScore * weight;
        }

        if (finalConfidence > confThreshold) {
          final centerX = detectionData[0]?.toDouble() ?? 0.0;
          final centerY = detectionData[1]?.toDouble() ?? 0.0;
          final width = detectionData[2]?.toDouble() ?? 0.0;
          final height = detectionData[3]?.toDouble() ?? 0.0;

          // Convert from tile coordinates to global coordinates
          final globalCenterX = (centerX * tileSize / tileSize) + tileOffsetX;
          final globalCenterY = (centerY * tileSize / tileSize) + tileOffsetY;
          final globalWidth = width * originalWidth / tileSize;
          final globalHeight = height * originalHeight / tileSize;

          final left = globalCenterX - globalWidth / 2;
          final top = globalCenterY - globalHeight / 2;

          if (globalWidth > 0 && globalHeight > 0) {
            detections.add(Detection(
              left: left,
              top: top,
              width: globalWidth,
              height: globalHeight,
              confidence: finalConfidence,
              classId: 0,
              label: 'room',
            ));
          }
        }
      } catch (e) {
        continue;
      }
    }
  }

  /// Process YOLOv8 format detections for tiles
  static void _processYOLOv8FormatTiled(
      dynamic output,
      int numFeatures,
      int numDetections,
      double tileOffsetX,
      double tileOffsetY,
      double originalWidth,
      double originalHeight,
      double weight,
      List<Detection> detections,
      double confThreshold) {
    for (int i = 0; i < numDetections; i++) {
      try {
        final batch = output[0];
        if (batch == null) continue;

        double confidence = 0.0;
        if (numFeatures > 4) {
          final featureArray = batch[4];
          if (featureArray != null && i < featureArray.length) {
            confidence = (featureArray[i]?.toDouble() ?? 0.0) * weight;
          }
        } else {
          confidence = 0.8 * weight;
        }

        if (confidence > confThreshold) {
          final centerX = batch[0]?[i]?.toDouble() ?? 0.0;
          final centerY = batch[1]?[i]?.toDouble() ?? 0.0;
          final width = batch[2]?[i]?.toDouble() ?? 0.0;
          final height = batch[3]?[i]?.toDouble() ?? 0.0;

          // Convert from tile coordinates to global coordinates
          final globalCenterX = (centerX * tileSize / tileSize) + tileOffsetX;
          final globalCenterY = (centerY * tileSize / tileSize) + tileOffsetY;
          final globalWidth = width * originalWidth / tileSize;
          final globalHeight = height * originalHeight / tileSize;

          final left = globalCenterX - globalWidth / 2;
          final top = globalCenterY - globalHeight / 2;

          if (globalWidth > 0 && globalHeight > 0) {
            detections.add(Detection(
              left: left,
              top: top,
              width: globalWidth,
              height: globalHeight,
              confidence: confidence,
              classId: 0,
              label: 'room',
            ));
          }
        }
      } catch (e) {
        continue;
      }
    }
  }
}

/// Data class for tile information
class TileData {
  final img.Image image;
  final int offsetX;
  final int offsetY;
  final int originalWidth;
  final int originalHeight;

  TileData({
    required this.image,
    required this.offsetX,
    required this.offsetY,
    required this.originalWidth,
    required this.originalHeight,
  });
}
