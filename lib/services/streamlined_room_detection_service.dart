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
  static const double confidenceThreshold =
      0.1; // Lowered from 0.3 to catch more detections
  static const double iouGroupingThreshold = 0.5;
  static const double nmsThreshold = 0.45;

  // Model only accepts 'room' as a class
  static const List<String> roomClassNames = [
    'room', // Single class model
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
        _log(
            '  - Model confidence threshold too low (current: $confidenceThreshold)');
        _log('  - Image doesn\'t contain recognizable room features');
        _log('  - Model not suitable for this type of floorplan');
        _log('  - Single-class model may need different preprocessing');
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
      final preprocessedData = await compute(_preprocessImage, resizedImage);
      _log('✓ Image preprocessing completed');

      List<Detection> allDetections = [];

      // Run inference on each model
      for (int i = 0; i < _interpreters.length; i++) {
        final interpreter = _interpreters[i];
        final weight = modelConfigs[i].weight;

        _log('Running inference on model ${i + 1}/${_interpreters.length}...');

        try {
          // Get input and output tensor information
          final inputTensor = interpreter.getInputTensor(0);
          final outputTensor = interpreter.getOutputTensor(0);

          _log('Model ${i + 1} input shape: ${inputTensor.shape}');
          _log('Model ${i + 1} input type: ${inputTensor.type}');
          _log('Model ${i + 1} output shape: ${outputTensor.shape}');
          _log('Model ${i + 1} output type: ${outputTensor.type}');

          // Try different input formats based on what the model expects
          _log('Attempting inference with different input formats...');

          // Create output buffer
          final output = _createOutputBuffer(outputTensor.shape);

          // Run inference with error handling for different input formats
          bool inferenceSuccess = false;
          String errorMessage = '';

          // Try Float32List format first (most efficient)
          try {
            final inputData = preprocessedData['float32List'] as Float32List;
            _log('Trying Float32List format (${inputData.length} elements)...');
            interpreter.run(inputData, output);
            _log(
                '✓ Model ${i + 1} inference completed with Float32List format');
            inferenceSuccess = true;
          } catch (e) {
            errorMessage = 'Float32List: $e';
            _log('Float32List format failed: $e');

            // Try nested list format as fallback
            try {
              final inputData = preprocessedData['nestedList'];
              _log('Trying nested list format...');
              interpreter.run(inputData, output);
              _log(
                  '✓ Model ${i + 1} inference completed with nested list format');
              inferenceSuccess = true;
            } catch (e2) {
              errorMessage += ', Nested list: $e2';
              _log('Nested list format also failed: $e2');

              // Try with different tensor input approach
              try {
                final inputData =
                    preprocessedData['float32List'] as Float32List;
                _log('Trying alternative tensor input...');
                // Reshape the data into proper tensor format
                final reshapedInput = [inputData];
                interpreter.run(reshapedInput, output);
                _log(
                    '✓ Model ${i + 1} inference completed with reshaped tensor');
                inferenceSuccess = true;
              } catch (e3) {
                errorMessage += ', Reshaped: $e3';
                _log(
                    'All input formats failed for model ${i + 1}: $errorMessage');
                // Don't throw exception here, continue with next model
              }
            }
          }

          if (!inferenceSuccess) {
            _log(
                '✗ Model ${i + 1} inference failed with all formats: $errorMessage');
            continue; // Skip this model and try the next one
          }

          // Verify output is not null
          if (output == null) {
            _log('✗ Model ${i + 1} produced null output');
            continue;
          }

          // Post-process detections in isolate
          final modelDetections = await compute(_postprocessDetections, {
            'output': output,
            'outputShape': outputTensor.shape,
            'originalWidth': originalImage.width.toDouble(),
            'originalHeight': originalImage.height.toDouble(),
            'weight': weight,
            'confidenceThreshold': confidenceThreshold,
          });

          _log('Model ${i + 1} produced ${modelDetections.length} detections');
          allDetections.addAll(modelDetections);
        } catch (e, stackTrace) {
          _log('✗ Model ${i + 1} inference failed: $e');
          _log(
              'Stack trace: ${stackTrace.toString().split('\n').take(3).join('\n')}');
          // Continue with other models
        }
      }

      // Apply NMS and final processing
      _log('Applying non-maximum suppression...');
      _log('Total raw detections before NMS: ${allDetections.length}');

      if (allDetections.isEmpty) {
        _log('⚠️ No detections found from any model');
        return <Detection>[]; // Return empty list instead of throwing
      }

      final averagedDetections = _averageOverlappingBoxes(allDetections);
      _log('Detections after averaging: ${averagedDetections.length}');
      final finalDetections = _applyNMS(averagedDetections);
      _log(
          '✓ Final processing completed: ${finalDetections.length} detections');

      return finalDetections;
    } catch (e, stackTrace) {
      _log('✗ Inference failed: $e');
      _log(
          'Full stack trace: ${stackTrace.toString().split('\n').take(5).join('\n')}');
      // Return empty list instead of crashing the app
      return <Detection>[];
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
  static dynamic _preprocessImage(img.Image image) {
    try {
      print('Preprocessing image: ${image.width}x${image.height}');

      // Ensure we have a valid 640x640 image
      if (image.width != inputSize || image.height != inputSize) {
        throw Exception(
            'Image must be ${inputSize}x$inputSize, got ${image.width}x${image.height}');
      }

      // Create different input formats for TensorFlow Lite compatibility
      final inputBytes = Float32List(1 * inputSize * inputSize * 3);
      int pixelIndex = 0;

      // Convert image to NHWC format (batch, height, width, channels)
      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = image.getPixel(x, y);

          // Normalize pixel values to [0, 1] range
          final r = pixel.r / 255.0;
          final g = pixel.g / 255.0;
          final b = pixel.b / 255.0;

          inputBytes[pixelIndex++] = r;
          inputBytes[pixelIndex++] = g;
          inputBytes[pixelIndex++] = b;
        }
      }

      print('Generated Float32List with ${inputBytes.length} elements');

      // Create nested list format (4D: [batch, height, width, channels])
      final nestedList = List.generate(1, (batch) {
        return List.generate(inputSize, (h) {
          return List.generate(inputSize, (w) {
            final baseIndex = (h * inputSize + w) * 3;
            return [
              inputBytes[baseIndex], // R
              inputBytes[baseIndex + 1], // G
              inputBytes[baseIndex + 2], // B
            ];
          });
        });
      });

      print('Generated nested list with shape: [1, $inputSize, $inputSize, 3]');

      return {
        'float32List': inputBytes,
        'nestedList': nestedList,
      };
    } catch (e, stackTrace) {
      print('Error in preprocessing: $e');
      print('Stack trace: $stackTrace');
      rethrow;
    }
  }

  /// Post-process detections in isolate
  static List<Detection> _postprocessDetections(Map<String, dynamic> params) {
    final output = params['output'];
    final outputShape = params['outputShape'] as List<int>;
    final originalWidth = params['originalWidth'] as double;
    final originalHeight = params['originalHeight'] as double;
    final weight = params['weight'] as double;
    final confThreshold = params['confidenceThreshold'] as double;

    List<Detection> detections = [];

    try {
      print('Postprocessing: Output shape = $outputShape');

      if (outputShape.length == 3) {
        // Handle both YOLOv5 [1, num_detections, num_features] and YOLOv8 transposed [1, num_features, num_detections]
        final dim1 = outputShape[1]; // Could be detections or features
        final dim2 = outputShape[2]; // Could be features or detections

        print('Postprocessing: dim1=$dim1, dim2=$dim2');

        if (dim1 > dim2) {
          // YOLOv5 format: [1, 25200, 85] - more detections than features
          print('Processing as YOLOv5 format');
          _processYOLOv5Format(output, dim1, dim2, originalWidth,
              originalHeight, weight, detections, confThreshold);
        } else {
          // YOLOv8 format: [1, 37, 8400] - more detections than features (transposed)
          print('Processing as YOLOv8 format (single-class room model)');
          _processYOLOv8Format(output, dim1, dim2, originalWidth,
              originalHeight, weight, detections, confThreshold);
        }
      } else if (outputShape.length == 2) {
        // Direct 2D format: [num_features, num_detections]
        final numFeatures = outputShape[0];
        final numDetections = outputShape[1];
        print('Processing as 2D YOLOv8 format');
        _processYOLOv8Format(output, numFeatures, numDetections, originalWidth,
            originalHeight, weight, detections, confThreshold);
      } else {
        throw Exception('Unsupported output shape: $outputShape');
      }

      print('Postprocessing complete: Found ${detections.length} detections');
    } catch (e) {
      print('Error in postprocessing: $e');
      // Return empty list instead of crashing
    }

    return detections;
  }

  /// Process YOLOv5 format detections
  static void _processYOLOv5Format(
      dynamic output,
      int numDetections,
      int numFeatures,
      double originalWidth,
      double originalHeight,
      double weight,
      List<Detection> detections,
      double confThreshold) {
    for (int i = 0; i < numDetections; i++) {
      try {
        // Safely access output values with null checks
        final detection = output[0]?[i];
        if (detection == null || detection.length < 5) continue;

        final objectness = detection[4]?.toDouble() ?? 0.0;

        if (objectness * weight > confThreshold) {
          // For single-class model, use objectness as confidence or check if there's a class score
          double finalConfidence = objectness * weight;

          // If there are class scores beyond index 5, use the first (and likely only) class
          if (numFeatures > 5 && detection.length > 5) {
            final classScore = detection[5]?.toDouble() ??
                1.0; // Default to 1.0 for single class
            finalConfidence = objectness * classScore * weight;
          }

          if (finalConfidence > confThreshold) {
            // Convert from center format to corner format
            final centerX = detection[0]?.toDouble() ?? 0.0;
            final centerY = detection[1]?.toDouble() ?? 0.0;
            final width = detection[2]?.toDouble() ?? 0.0;
            final height = detection[3]?.toDouble() ?? 0.0;

            final left = (centerX - width / 2) * originalWidth / inputSize;
            final top = (centerY - height / 2) * originalHeight / inputSize;
            final detWidth = width * originalWidth / inputSize;
            final detHeight = height * originalHeight / inputSize;

            // Skip invalid detections
            if (detWidth <= 0 || detHeight <= 0) continue;

            detections.add(Detection(
              left: left,
              top: top,
              width: detWidth,
              height: detHeight,
              confidence: finalConfidence,
              classId: 0, // Single class always has ID 0
              label: 'room', // Always 'room' for single-class model
            ));
          }
        }
      } catch (e) {
        // Skip this detection and continue
        continue;
      }
    }
  }

  /// Process YOLOv8 format detections
  static void _processYOLOv8Format(
      dynamic output,
      int numFeatures,
      int numDetections,
      double originalWidth,
      double originalHeight,
      double weight,
      List<Detection> detections,
      double confThreshold) {
    for (int i = 0; i < numDetections; i++) {
      try {
        // Safely access output values with null checks
        final batch = output[0];
        if (batch == null) continue;

        // For single-class model, the class score is at feature index 4
        double confidence = 0.0;

        if (numFeatures > 4) {
          // Extract the single class score (feature 4)
          final featureArray = batch[4];
          if (featureArray != null && i < featureArray.length) {
            confidence = (featureArray[i]?.toDouble() ?? 0.0) * weight;
          }
        } else {
          // If no class scores, use a default high confidence for detected objects
          confidence = 0.8 * weight;
        }

        if (confidence > confThreshold) {
          // Extract bounding box coordinates
          final centerX = batch[0]?[i]?.toDouble() ?? 0.0;
          final centerY = batch[1]?[i]?.toDouble() ?? 0.0;
          final width = batch[2]?[i]?.toDouble() ?? 0.0;
          final height = batch[3]?[i]?.toDouble() ?? 0.0;

          final left = (centerX - width / 2) * originalWidth / inputSize;
          final top = (centerY - height / 2) * originalHeight / inputSize;
          final detWidth = width * originalWidth / inputSize;
          final detHeight = height * originalHeight / inputSize;

          // Skip invalid detections
          if (detWidth <= 0 || detHeight <= 0) continue;

          detections.add(Detection(
            left: left,
            top: top,
            width: detWidth,
            height: detHeight,
            confidence: confidence,
            classId: 0, // Single class always has ID 0
            label: 'room', // Always 'room' for single-class model
          ));
        }
      } catch (e) {
        // Skip this detection and continue
        continue;
      }
    }
  }

  /// Create output buffer based on model output shape
  dynamic _createOutputBuffer(List<int> shape) {
    if (shape.length == 3) {
      // YOLOv5 format: [batch, detections, features] or YOLOv8: [batch, features, detections]
      return List.generate(shape[0],
          (b) => List.generate(shape[1], (d) => List.filled(shape[2], 0.0)));
    } else if (shape.length == 2) {
      // 2D format: [features, detections]
      return List.generate(shape[0], (f) => List.filled(shape[1], 0.0));
    } else if (shape.length == 4) {
      // 4D format: [batch, height, width, channels]
      return List.generate(
          shape[0],
          (b) => List.generate(
              shape[1],
              (h) =>
                  List.generate(shape[2], (w) => List.filled(shape[3], 0.0))));
    } else {
      throw Exception(
          'Unsupported output shape: $shape (${shape.length}D tensor)');
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
