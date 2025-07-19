import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:floorplan_detection_app/interfaces/detection.dart';
import 'package:floorplan_detection_app/interfaces/model_config.dart';
import 'package:floorplan_detection_app/services/pdf_processing_service.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

/// Streamlined room detection service without tiling
/// Processes entire images resized to 640x640 for direct model inference
class StreamlinedRoomDetectionService {
  static const int inputSize = 640;
  static const double confidenceThreshold = 0.3;
  static const double iouGroupingThreshold = 0.5;
  static const double nmsThreshold = 0.45;

  // Room-related class names
  static const List<String> roomClassNames = [
    'room',
    'bedroom',
    'living_room',
    'kitchen',
    'bathroom',
    'office',
    'dining_room'
  ];

  final List<ModelConfig> modelConfigs;
  final List<Interpreter> _interpreters = [];
  bool _isModelLoaded = false;
  Function(String)? onLog;

  StreamlinedRoomDetectionService({
    required this.modelConfigs,
    this.onLog,
  });

  /// Load all models asynchronously
  Future<bool> loadModels() async {
    try {
      _log('Loading ${modelConfigs.length} model(s)...');

      for (int i = 0; i < modelConfigs.length; i++) {
        final config = modelConfigs[i];
        _log('Loading model ${i + 1}: ${config.assetPath}');

        final interpreter = await Interpreter.fromAsset(config.assetPath);
        _interpreters.add(interpreter);

        _log('✓ Model ${i + 1} loaded successfully');
      }

      _isModelLoaded = true;
      _log('✓ All ${_interpreters.length} models loaded successfully');
      return true;
    } catch (e) {
      _log('✗ Failed to load models: $e');
      _isModelLoaded = false;
      return false;
    }
  }

  bool get isModelLoaded => _isModelLoaded;

  /// Process a PDF file by rendering it to a single 640x640 image
  Future<List<Detection>> processPDF(File pdfFile, {double dpi = 150.0}) async {
    try {
      _log('=== Starting PDF Processing ===');
      _log('Input PDF: ${pdfFile.path}');

      // Step 1: Convert PDF to image
      _log('Converting PDF to image (DPI: $dpi)...');
      final image = await PdfProcessingService.pdfToImage(pdfFile, dpi: dpi);
      if (image == null) {
        throw Exception('Failed to convert PDF to image');
      }

      _log('✓ PDF converted to image: ${image.width}x${image.height} pixels');

      // Step 2: Process the full image
      return await processFullImage(image);
    } catch (e) {
      _log('✗ Error processing PDF: $e');
      rethrow;
    }
  }

  /// Process a regular image file
  Future<List<Detection>> processImageFile(File imageFile) async {
    try {
      _log('=== Starting Image Processing ===');
      _log('Input image: ${imageFile.path}');

      // Step 1: Load and decode image
      _log('Loading and decoding image...');
      final imageBytes = await imageFile.readAsBytes();
      _log('Image file size: ${imageBytes.length} bytes');

      final image = await compute(_decodeImage, imageBytes);
      if (image == null) {
        throw Exception('Failed to decode image');
      }

      _log('✓ Image decoded: ${image.width}x${image.height} pixels');

      // Step 2: Process the full image
      return await processFullImage(image);
    } catch (e) {
      _log('✗ Error processing image: $e');
      rethrow;
    }
  }

  /// Process a full image by resizing to 640x640 and running inference
  Future<List<Detection>> processFullImage(img.Image image) async {
    try {
      _log('Processing full image: ${image.width}x${image.height} pixels');

      // Step 1: Resize image to 640x640 (regardless of aspect ratio)
      _log('Resizing image to ${inputSize}x$inputSize pixels...');
      final resizedImage = await compute(_resizeImage, {
        'image': image,
        'size': inputSize,
      });

      _log('✓ Image resized to ${resizedImage.width}x${resizedImage.height}');

      // Step 2: Run inference asynchronously
      _log('Starting model inference...');
      final detections = await _runInferenceAsync(resizedImage, image);

      _log('✓ Inference completed: ${detections.length} detections found');

      // Log detection results
      if (detections.isEmpty) {
        _log('⚠️ No rooms detected. This could be due to:');
        _log('  - Model confidence threshold too high');
        _log('  - Image doesn\'t contain recognizable room features');
        _log('  - Model not trained for this type of floorplan');
      } else {
        _log('=== Detection Results ===');
        for (int i = 0; i < detections.length; i++) {
          final detection = detections[i];
          _log(
              'Room ${i + 1}: ${detection.label} (${(detection.confidence * 100).toStringAsFixed(1)}% confidence)');
        }
      }

      return detections;
    } catch (e) {
      _log('✗ Error during inference: $e');
      rethrow;
    }
  }

  /// Run model inference asynchronously to prevent UI blocking
  Future<List<Detection>> _runInferenceAsync(
      img.Image resizedImage, img.Image originalImage) async {
    try {
      if (!_isModelLoaded || _interpreters.isEmpty) {
        throw Exception('Models not loaded');
      }

      // Preprocess image in isolate to avoid blocking UI
      _log('Preprocessing image for model input...');
      final inputData = await compute(_preprocessImage, resizedImage);
      _log('✓ Image preprocessing completed');

      List<Detection> allDetections = [];

      // Run inference on each model
      for (int i = 0; i < _interpreters.length; i++) {
        final interpreter = _interpreters[i];
        final weight = modelConfigs[i].weight;

        _log('Running inference on model ${i + 1}/${_interpreters.length}...');

        try {
          // Get output shape to determine model format
          final outputShape = interpreter.getOutputTensor(0).shape;
          _log('Model ${i + 1} output shape: $outputShape');

          // Create output buffer
          final output = _createOutputBuffer(outputShape);

          // Run inference
          interpreter.run(inputData, output);
          _log('✓ Model ${i + 1} inference completed');

          // Post-process detections in isolate
          final modelDetections = await compute(_postprocessDetections, {
            'output': output,
            'outputShape': outputShape,
            'originalWidth': originalImage.width.toDouble(),
            'originalHeight': originalImage.height.toDouble(),
            'weight': weight,
          });

          _log('Model ${i + 1} produced ${modelDetections.length} detections');
          allDetections.addAll(modelDetections);
        } catch (e) {
          _log('✗ Model ${i + 1} inference failed: $e');
          // Continue with other models
        }
      }

      // Apply NMS and final processing
      _log('Applying non-maximum suppression...');
      final finalDetections =
          _applyNMS(_averageOverlappingBoxes(allDetections));
      _log(
          '✓ Final processing completed: ${finalDetections.length} detections');

      return finalDetections;
    } catch (e) {
      _log('✗ Inference failed: $e');
      rethrow;
    }
  }

  /// Dispose resources
  void dispose() {
    _log('Disposing detection service resources...');
    for (final interpreter in _interpreters) {
      interpreter.close();
    }
    _interpreters.clear();
    _isModelLoaded = false;
  }

  /// Helper method for logging
  void _log(String message) {
    if (onLog != null) {
      onLog!(message);
    }
    print(message); // Also log to console
  }

  // Static methods for isolate processing

  /// Decode image in isolate
  static img.Image? _decodeImage(Uint8List imageBytes) {
    return img.decodeImage(imageBytes);
  }

  /// Resize image in isolate
  static img.Image _resizeImage(Map<String, dynamic> params) {
    final image = params['image'] as img.Image;
    final size = params['size'] as int;
    return img.copyResize(image, width: size, height: size);
  }

  /// Preprocess image for model input in isolate
  static List<List<List<List<double>>>> _preprocessImage(img.Image image) {
    final bytes = Float32List(inputSize * inputSize * 3);
    int pixelIndex = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = image.getPixel(x, y);
        bytes[pixelIndex++] = pixel.r / 255.0;
        bytes[pixelIndex++] = pixel.g / 255.0;
        bytes[pixelIndex++] = pixel.b / 255.0;
      }
    }

    return [
      List.generate(inputSize, (h) {
        return List.generate(inputSize, (w) {
          return List.generate(3, (c) => bytes[(h * inputSize + w) * 3 + c]);
        });
      })
    ];
  }

  /// Post-process detections in isolate
  static List<Detection> _postprocessDetections(Map<String, dynamic> params) {
    final output = params['output'];
    final outputShape = params['outputShape'] as List<int>;
    final originalWidth = params['originalWidth'] as double;
    final originalHeight = params['originalHeight'] as double;
    final weight = params['weight'] as double;

    List<Detection> detections = [];

    if (outputShape.length == 3) {
      // YOLOv5 format: [1, 25200, 85] or similar
      final numDetections = outputShape[1];
      final numFeatures = outputShape[2];

      for (int i = 0; i < numDetections; i++) {
        final confidence = output[0][i][4] * weight;

        if (confidence > confidenceThreshold) {
          // Extract class scores (skip first 5 elements: x, y, w, h, objectness)
          double maxClassScore = 0.0;
          int bestClassId = 0;

          for (int c = 5; c < numFeatures; c++) {
            final classScore = output[0][i][c];
            if (classScore > maxClassScore) {
              maxClassScore = classScore;
              bestClassId = c - 5;
            }
          }

          final finalConfidence = confidence * maxClassScore;
          if (finalConfidence > confidenceThreshold) {
            // Convert from center format to corner format
            final centerX = output[0][i][0];
            final centerY = output[0][i][1];
            final width = output[0][i][2];
            final height = output[0][i][3];

            final left = (centerX - width / 2) * originalWidth / inputSize;
            final top = (centerY - height / 2) * originalHeight / inputSize;
            final detWidth = width * originalWidth / inputSize;
            final detHeight = height * originalHeight / inputSize;

            final className = bestClassId < roomClassNames.length
                ? roomClassNames[bestClassId]
                : 'room';

            detections.add(Detection(
              left: left,
              top: top,
              width: detWidth,
              height: detHeight,
              confidence: finalConfidence,
              classId: bestClassId,
              label: className,
            ));
          }
        }
      }
    } else if (outputShape.length == 2) {
      // YOLOv8 format: [1, 84, 8400] or similar
      final numFeatures = outputShape[1];
      final numDetections = outputShape[2];

      for (int i = 0; i < numDetections; i++) {
        // Extract class scores (features 4 onwards)
        double maxClassScore = 0.0;
        int bestClassId = 0;

        for (int c = 4; c < numFeatures; c++) {
          final classScore = output[0][c][i];
          if (classScore > maxClassScore) {
            maxClassScore = classScore;
            bestClassId = c - 4;
          }
        }

        final confidence = maxClassScore * weight;
        if (confidence > confidenceThreshold) {
          // Convert from center format to corner format
          final centerX = output[0][0][i];
          final centerY = output[0][1][i];
          final width = output[0][2][i];
          final height = output[0][3][i];

          final left = (centerX - width / 2) * originalWidth / inputSize;
          final top = (centerY - height / 2) * originalHeight / inputSize;
          final detWidth = width * originalWidth / inputSize;
          final detHeight = height * originalHeight / inputSize;

          final className = bestClassId < roomClassNames.length
              ? roomClassNames[bestClassId]
              : 'room';

          detections.add(Detection(
            left: left,
            top: top,
            width: detWidth,
            height: detHeight,
            confidence: confidence,
            classId: bestClassId,
            label: className,
          ));
        }
      }
    }

    return detections;
  }

  /// Create output buffer based on model output shape
  dynamic _createOutputBuffer(List<int> shape) {
    if (shape.length == 3) {
      // YOLOv5 format: [batch, detections, features]
      return List.generate(shape[0],
          (b) => List.generate(shape[1], (d) => List.filled(shape[2], 0.0)));
    } else if (shape.length == 2) {
      // YOLOv8 format: [batch, features, detections]
      return List.generate(shape[0],
          (b) => List.generate(shape[1], (f) => List.filled(shape[2], 0.0)));
    } else {
      throw Exception('Unsupported output shape: $shape');
    }
  }

  /// Average overlapping boxes
  List<Detection> _averageOverlappingBoxes(List<Detection> detections) {
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
        if (iou > iouGroupingThreshold) {
          group.add(detections[j]);
          used[j] = true;
        }
      }

      // Average the group
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

  /// Apply Non-Maximum Suppression
  List<Detection> _applyNMS(List<Detection> detections) {
    if (detections.isEmpty) return detections;

    // Sort by confidence
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));

    List<Detection> kept = [];
    List<bool> suppressed = List.filled(detections.length, false);

    for (int i = 0; i < detections.length; i++) {
      if (suppressed[i]) continue;

      kept.add(detections[i]);

      for (int j = i + 1; j < detections.length; j++) {
        if (suppressed[j]) continue;

        final iou = _calculateIoU(detections[i], detections[j]);
        if (iou > nmsThreshold) {
          suppressed[j] = true;
        }
      }
    }

    return kept;
  }

  /// Calculate Intersection over Union
  double _calculateIoU(Detection a, Detection b) {
    final left = [a.left, b.left].reduce((a, b) => a > b ? a : b);
    final top = [a.top, b.top].reduce((a, b) => a > b ? a : b);
    final right =
        [a.left + a.width, b.left + b.width].reduce((a, b) => a < b ? a : b);
    final bottom =
        [a.top + a.height, b.top + b.height].reduce((a, b) => a < b ? a : b);

    if (right <= left || bottom <= top) return 0.0;

    final intersection = (right - left) * (bottom - top);
    final areaA = a.width * a.height;
    final areaB = b.width * b.height;
    final union = areaA + areaB - intersection;

    return intersection / union;
  }
}
