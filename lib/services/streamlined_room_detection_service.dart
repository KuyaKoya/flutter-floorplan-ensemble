import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:floorplan_detection_app/interfaces/detection.dart';
import 'package:floorplan_detection_app/services/pdf_processing_service.dart';
import 'package:floorplan_detection_app/services/base_room_detection_service.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class StreamlinedRoomDetectionService extends BaseRoomDetectionService {
  static const int inputSize = 640;
  static const double confidenceThreshold = 0.1;
  static const double iouGroupingThreshold = 0.5;
  static const double nmsThreshold = 0.45;
  static const double minVarianceThreshold = 10.0; // Minimum pixel variance

  static const List<String> roomClassNames = ['room'];

  final List<Interpreter> _interpreters = [];
  bool _isModelLoaded = false;

  StreamlinedRoomDetectionService({
    required super.modelConfigs,
    super.onLog,
  });

  @override
  Future<bool> loadModels() async {
    try {
      logMessage('Loading ${modelConfigs.length} model(s)...');

      for (int i = 0; i < modelConfigs.length; i++) {
        final config = modelConfigs[i];
        logMessage('Loading model ${i + 1}: ${config.assetPath}');

        final interpreter = await Interpreter.fromAsset(config.assetPath);
        _interpreters.add(interpreter);

        logMessage('✓ Model ${i + 1} loaded successfully');
      }

      _isModelLoaded = true;
      logMessage('✓ All ${_interpreters.length} models loaded successfully');
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
      logMessage('=== Starting PDF Processing ===');
      logMessage('Input PDF: ${pdfFile.path}');

      logMessage('Converting PDF to image (DPI: $dpi)...');
      final image = await PdfProcessingService.pdfToImage(pdfFile, dpi: dpi);
      if (image == null) {
        throw Exception('Failed to convert PDF to image');
      }

      logMessage(
          '✓ PDF converted to image: ${image.width}x${image.height} pixels');
      return await processFullImage(image);
    } catch (e) {
      logMessage('✗ Error processing PDF: $e');
      rethrow;
    }
  }

  @override
  Future<List<Detection>> processImageFile(File imageFile) async {
    try {
      logMessage('=== Starting Image Processing ===');
      logMessage('Input image: ${imageFile.path}');

      logMessage('Loading and decoding image...');
      final imageBytes = await imageFile.readAsBytes();
      logMessage('Image file size: ${imageBytes.length} bytes');

      final image = await compute(_decodeImage, imageBytes);
      if (image == null) {
        throw Exception('Failed to decode image');
      }

      logMessage('✓ Image decoded: ${image.width}x${image.height} pixels');
      return await processFullImage(image);
    } catch (e) {
      logMessage('✗ Error processing image: $e');
      rethrow;
    }
  }

  Future<List<Detection>> processFullImage(img.Image image) async {
    try {
      logMessage(
          'Processing full image: ${image.width}x${image.height} pixels');

      logMessage('Resizing image to ${inputSize}x$inputSize pixels...');
      final resizedImage = await compute(_resizeImage, {
        'image': image,
        'size': inputSize,
      });

      logMessage(
          '✓ Image resized to ${resizedImage.width}x${resizedImage.height}');

      // Validate image content with a more comprehensive sample
      final pixelSample = resizedImage.getPixel(0, 0);
      logMessage(
          'Sample pixel (0,0): R=${pixelSample.r}, G=${pixelSample.g}, B=${pixelSample.b}');

      // Sample pixels from multiple regions to get a better representation
      double pixelVariance = 0.0;
      double pixelMean = 0.0;
      int pixelCount = 0;

      // Sample from 4 corners and center region (50x50 each)
      final sampleSize = 50;
      final regions = [
        {'x': 0, 'y': 0}, // top-left
        {'x': resizedImage.width - sampleSize, 'y': 0}, // top-right
        {'x': 0, 'y': resizedImage.height - sampleSize}, // bottom-left
        {
          'x': resizedImage.width - sampleSize,
          'y': resizedImage.height - sampleSize
        }, // bottom-right
        {
          'x': (resizedImage.width - sampleSize) ~/ 2,
          'y': (resizedImage.height - sampleSize) ~/ 2
        }, // center
      ];

      // Calculate mean first
      for (final region in regions) {
        int startX = region['x']!.clamp(0, resizedImage.width - 1);
        int startY = region['y']!.clamp(0, resizedImage.height - 1);
        int endX = (startX + sampleSize).clamp(0, resizedImage.width);
        int endY = (startY + sampleSize).clamp(0, resizedImage.height);

        for (int y = startY; y < endY; y++) {
          for (int x = startX; x < endX; x++) {
            final pixel = resizedImage.getPixel(x, y);
            pixelMean += pixel.r;
            pixelCount++;
          }
        }
      }
      pixelMean /= pixelCount;

      // Calculate variance
      for (final region in regions) {
        int startX = region['x']!.clamp(0, resizedImage.width - 1);
        int startY = region['y']!.clamp(0, resizedImage.height - 1);
        int endX = (startX + sampleSize).clamp(0, resizedImage.width);
        int endY = (startY + sampleSize).clamp(0, resizedImage.height);

        for (int y = startY; y < endY; y++) {
          for (int x = startX; x < endX; x++) {
            final pixel = resizedImage.getPixel(x, y);
            pixelVariance += (pixel.r - pixelMean) * (pixel.r - pixelMean);
          }
        }
      }
      pixelVariance /= pixelCount;

      logMessage(
          'Pixel mean: ${pixelMean.toStringAsFixed(2)}, variance: ${pixelVariance.toStringAsFixed(2)} (sampled ${pixelCount} pixels from 5 regions)');
      if (pixelVariance < minVarianceThreshold) {
        logMessage(
            '✗ Image rejected: Pixel variance (${pixelVariance.toStringAsFixed(2)}) below threshold ($minVarianceThreshold). Image appears too uniform.');
        throw Exception(
            'Image appears too uniform. Please ensure the image contains a floorplan with clear room boundaries and sufficient contrast.');
      }

      logMessage('Starting model inference...');
      final detections = await _runInferenceAsync(resizedImage, image);

      logMessage(
          '✓ Inference completed: ${detections.length} detections found');

      if (detections.isEmpty) {
        logMessage('⚠️ No rooms detected. This could be due to:');
        logMessage(
            '  - Model confidence threshold too low (current: $confidenceThreshold)');
        logMessage('  - Image doesn\'t contain recognizable room features');
        logMessage('  - Model not trained for this type of floorplan');
        logMessage(
            '  - Preprocessing mismatch (expected grayscale with adaptive thresholding)');
      } else {
        logMessage('=== Detection Results ===');
        for (int i = 0; i < detections.length; i++) {
          final detection = detections[i];
          logMessage(
              'Room ${i + 1}: ${detection.label} (${(detection.confidence * 100).toStringAsFixed(1)}% confidence)');
        }
      }

      return detections;
    } catch (e) {
      logMessage('✗ Error during inference: $e');
      rethrow;
    }
  }

  Future<List<Detection>> _runInferenceAsync(
      img.Image resizedImage, img.Image originalImage) async {
    try {
      if (!_isModelLoaded || _interpreters.isEmpty) {
        throw Exception('Models not loaded');
      }

      logMessage('Preprocessing image for model input...');
      final preprocessedData = await compute(_preprocessImage, resizedImage);
      logMessage('✓ Image preprocessing completed');
      logMessage(
          'Float32List length: ${preprocessedData['float32List'].length}');
      logMessage(
          'Input data sample (100 values): ${preprocessedData['float32List'].sublist(0, 100)}');
      logMessage('Nested list shape: ${preprocessedData['nestedList'].length}x'
          '${preprocessedData['nestedList'][0].length}x'
          '${preprocessedData['nestedList'][0][0].length}x'
          '${preprocessedData['nestedList'][0][0][0].length}');

      // Validate input data - lower threshold and warning instead of exception
      final inputData = preprocessedData['float32List'] as Float32List;
      double inputMean = inputData.reduce((a, b) => a + b) / inputData.length;
      double inputVariance = inputData.fold(
              0.0, (sum, val) => sum + (val - inputMean) * (val - inputMean)) /
          inputData.length;
      logMessage(
          'Input data mean: ${inputMean.toStringAsFixed(2)}, variance: ${inputVariance.toStringAsFixed(4)}');
      if (inputVariance < 0.001) {
        logMessage(
            '⚠️ Warning: Input data variance (${inputVariance.toStringAsFixed(4)}) is very low. Image may be too uniform, but continuing with inference...');
      }

      List<Detection> allDetections = [];

      for (int i = 0; i < _interpreters.length; i++) {
        final interpreter = _interpreters[i];
        final weight = modelConfigs[i].weight;

        logMessage(
            'Running inference on model ${i + 1}/${_interpreters.length}...');

        try {
          final inputTensor = interpreter.getInputTensor(0);
          final outputTensors = interpreter.getOutputTensors();

          logMessage(
              'Model ${i + 1} input shape: ${inputTensor.shape}, type: ${inputTensor.type}');
          for (int j = 0; j < outputTensors.length; j++) {
            logMessage(
                'Model ${i + 1} output $j shape: ${outputTensors[j].shape}, type: ${outputTensors[j].type}');
          }

          final outputs = _createOutputBuffers(interpreter);
          final hasSegmentation = outputTensors.length > 1;

          try {
            if (inputData.length != 1 * inputSize * inputSize * 3) {
              throw Exception(
                  'Invalid Float32List length: ${inputData.length}, expected ${1 * inputSize * inputSize * 3}');
            }

            interpreter.allocateTensors();
            logMessage('Tensors allocated successfully for model ${i + 1}');

            // Reshape the input data to match the expected input tensor shape [1, 640, 640, 3]
            final reshapedInput = preprocessedData['nestedList'];

            final outputMap = <int, Object>{};
            for (int j = 0; j < outputs.length; j++) {
              outputMap[j] = outputs[j];
            }
            interpreter.runForMultipleInputs([reshapedInput], outputMap);

            logMessage(
                '✓ Model ${i + 1} inference completed with nested list format');
          } catch (e, stackTrace) {
            logMessage('✗ Model ${i + 1} inference failed: $e');
            logMessage(
                'Stack trace: ${stackTrace.toString().split('\n').take(3).join('\n')}');
            continue;
          }

          dynamic maskOutput =
              hasSegmentation && outputs.length > 1 ? outputs[1] : null;
          if (hasSegmentation && outputTensors.length > 1) {
            final maskTensor = outputTensors[1];
            logMessage('Segmentation output shape: ${maskTensor.shape}');
          }

          final modelDetections = await compute(_postprocessDetections, {
            'output': outputs[0],
            'maskOutput': maskOutput,
            'outputShape': outputTensors[0].shape,
            'originalWidth': originalImage.width.toDouble(),
            'originalHeight': originalImage.height.toDouble(),
            'weight': weight,
            'confidenceThreshold': confidenceThreshold,
            'hasSegmentation': hasSegmentation,
          });

          logMessage(
              'Model ${i + 1} produced ${modelDetections.length} detections');
          allDetections.addAll(modelDetections);
        } catch (e, stackTrace) {
          logMessage('✗ Model ${i + 1} inference failed: $e');
          logMessage(
              'Stack trace: ${stackTrace.toString().split('\n').take(3).join('\n')}');
        }
      }

      logMessage('Applying non-maximum suppression...');
      logMessage('Total raw detections before NMS: ${allDetections.length}');

      if (allDetections.isEmpty) {
        logMessage('⚠️ No detections found from any model');
        return <Detection>[];
      }

      final averagedDetections = _averageOverlappingBoxes(allDetections);
      logMessage('Detections after averaging: ${averagedDetections.length}');
      final finalDetections = _applyNMS(averagedDetections);
      logMessage(
          '✓ Final processing completed: ${finalDetections.length} detections');

      return finalDetections;
    } catch (e, stackTrace) {
      logMessage('✗ Inference failed: $e');
      logMessage(
          'Stack trace: ${stackTrace.toString().split('\n').take(5).join('\n')}');
      return <Detection>[];
    }
  }

  @override
  void dispose() {
    logMessage('Disposing detection service resources...');
    for (final interpreter in _interpreters) {
      interpreter.close();
    }
    _interpreters.clear();
    _isModelLoaded = false;
  }

  static img.Image? _decodeImage(Uint8List imageBytes) {
    return img.decodeImage(imageBytes);
  }

  static img.Image _resizeImage(Map<String, dynamic> params) {
    final image = params['image'] as img.Image;
    final size = params['size'] as int;
    return img.copyResize(image, width: size, height: size);
  }

  static dynamic _preprocessImage(img.Image image) {
    try {
      print('Preprocessing image: ${image.width}x${image.height}');

      if (image.width != inputSize || image.height != inputSize) {
        throw Exception(
            'Image must be ${inputSize}x$inputSize, got ${image.width}x${image.height}');
      }

      // Convert to grayscale
      final grayscaleImage = img.grayscale(image);

      // Try adaptive thresholding first
      img.Image processedImage;
      try {
        final thresholdedImage = _adaptiveThreshold(grayscaleImage);

        // Check if adaptive thresholding made the image too uniform
        final sampleVariance = _calculateImageVariance(thresholdedImage);
        if (sampleVariance > 0.1) {
          // Adaptive thresholding worked well
          processedImage = img.adjustColor(thresholdedImage, contrast: 1.5);
          print(
              'Using adaptive thresholded image (variance: ${sampleVariance.toStringAsFixed(4)})');
        } else {
          // Fallback: use enhanced grayscale without thresholding
          processedImage =
              img.adjustColor(grayscaleImage, contrast: 2.0, brightness: 0.1);
          print(
              'Fallback: using enhanced grayscale (adaptive threshold variance too low: ${sampleVariance.toStringAsFixed(4)})');
        }
      } catch (e) {
        // Fallback: use enhanced grayscale without thresholding
        processedImage =
            img.adjustColor(grayscaleImage, contrast: 2.0, brightness: 0.1);
        print(
            'Fallback: using enhanced grayscale due to adaptive threshold error: $e');
      }

      final inputBytes = Float32List(1 * inputSize * inputSize * 3);
      int pixelIndex = 0;

      for (int y = 0; y < inputSize; y++) {
        for (int x = 0; x < inputSize; x++) {
          final pixel = processedImage.getPixel(x, y);
          final value = pixel.r / 255.0; // Grayscale: R=G=B
          inputBytes[pixelIndex++] = value;
          inputBytes[pixelIndex++] = value;
          inputBytes[pixelIndex++] = value;
        }
      }

      print('Generated Float32List with ${inputBytes.length} elements');
      print('Input data sample (100 values): ${inputBytes.sublist(0, 100)}');

      // Validate input data - use a lower threshold for preprocessed data
      double inputMean = inputBytes.reduce((a, b) => a + b) / inputBytes.length;
      double inputVariance = inputBytes.fold(
              0.0, (sum, val) => sum + (val - inputMean) * (val - inputMean)) /
          inputBytes.length;
      print(
          'Input data mean: ${inputMean.toStringAsFixed(2)}, variance: ${inputVariance.toStringAsFixed(2)}');
      if (inputVariance < 0.001) {
        // Lower threshold for preprocessed data
        print(
            '✗ Preprocessed image has very low variance (${inputVariance.toStringAsFixed(4)}). Image may lack sufficient contrast after preprocessing.');
        // Don't throw exception here, let the model try to process it
        print(
            '⚠️ Warning: Low variance detected, but continuing with inference...');
      }

      final nestedList = List.generate(1, (batch) {
        return List.generate(inputSize, (h) {
          return List.generate(inputSize, (w) {
            final baseIndex = (h * inputSize + w) * 3;
            return [
              inputBytes[baseIndex],
              inputBytes[baseIndex + 1],
              inputBytes[baseIndex + 2],
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

  static img.Image _adaptiveThreshold(img.Image image) {
    // Implement adaptive thresholding to enhance contrast
    final thresholded = img.Image.from(image);
    final windowSize = 15; // Adjust based on image characteristics
    final c = 7; // Constant subtracted from mean

    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        // Compute local mean in a window
        double sum = 0.0;
        int count = 0;
        for (int dy = -windowSize ~/ 2; dy <= windowSize ~/ 2; dy++) {
          for (int dx = -windowSize ~/ 2; dx <= windowSize ~/ 2; dx++) {
            int ny = (y + dy).clamp(0, image.height - 1);
            int nx = (x + dx).clamp(0, image.width - 1);
            sum += image.getPixel(nx, ny).r;
            count++;
          }
        }
        final localMean = sum / count;
        final pixel = image.getPixel(x, y).r;
        thresholded.setPixel(
            x, y, (pixel > localMean - c ? 255 : 0) as img.Color);
      }
    }
    return thresholded;
  }

  static List<Detection> _postprocessDetections(Map<String, dynamic> params) {
    final output = params['output'];
    final maskOutput = params['maskOutput'];
    final outputShape = params['outputShape'] as List<int>;
    final originalWidth = params['originalWidth'] as double;
    final originalHeight = params['originalHeight'] as double;
    final weight = params['weight'] as double;
    final confThreshold = params['confidenceThreshold'] as double;
    final hasSegmentation = params['hasSegmentation'] as bool;

    List<Detection> detections = [];

    try {
      print('Postprocessing: Output shape = $outputShape');
      print('Has segmentation: $hasSegmentation');

      if (outputShape.length == 3) {
        final dim1 = outputShape[1];
        final dim2 = outputShape[2];

        print('Postprocessing: dim1=$dim1, dim2=$dim2');

        if (dim1 > dim2) {
          print('Processing as YOLOv5 format');
          _processYOLOv5Format(
              output,
              dim1,
              dim2,
              originalWidth,
              originalHeight,
              weight,
              detections,
              confThreshold,
              maskOutput,
              hasSegmentation);
        } else {
          print('Processing as YOLOv8 format (single-class room model)');
          _processYOLOv8Format(
              output,
              dim1,
              dim2,
              originalWidth,
              originalHeight,
              weight,
              detections,
              confThreshold,
              maskOutput,
              hasSegmentation);
        }
      } else if (outputShape.length == 2) {
        final numFeatures = outputShape[0];
        final numDetections = outputShape[1];
        print('Processing as 2D YOLOv8 format');
        _processYOLOv8Format(
            output,
            numFeatures,
            numDetections,
            originalWidth,
            originalHeight,
            weight,
            detections,
            confThreshold,
            maskOutput,
            hasSegmentation);
      } else {
        throw Exception('Unsupported output shape: $outputShape');
      }

      print('Postprocessing complete: Found ${detections.length} detections');
    } catch (e) {
      print('Error in postprocessing: $e');
    }

    return detections;
  }

  static void _processYOLOv5Format(
      dynamic output,
      int numDetections,
      int numFeatures,
      double originalWidth,
      double originalHeight,
      double weight,
      List<Detection> detections,
      double confThreshold,
      [dynamic maskOutput,
      bool hasSegmentation = false]) {
    for (int i = 0; i < numDetections; i++) {
      try {
        final detectionData = output[0]?[i];
        if (detectionData == null || detectionData.length < 5) continue;

        final objectness = detectionData[4]?.toDouble() ?? 0.0;

        if (objectness * weight > confThreshold) {
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

            final left = (centerX - width / 2) * originalWidth / inputSize;
            final top = (centerY - height / 2) * originalHeight / inputSize;
            final detWidth = width * originalWidth / inputSize;
            final detHeight = height * originalHeight / inputSize;

            if (detWidth <= 0 || detHeight <= 0) continue;

            Detection detection;

            if (hasSegmentation && maskOutput != null) {
              final mask = _extractAndResizeMask(
                  maskOutput, i, originalWidth.toInt(), originalHeight.toInt());
              detection = SegmentationDetection(
                left: left,
                top: top,
                width: detWidth,
                height: detHeight,
                confidence: finalConfidence,
                classId: 0,
                label: 'room',
                mask: mask,
              );
            } else {
              detection = Detection(
                left: left,
                top: top,
                width: detWidth,
                height: detHeight,
                confidence: finalConfidence,
                classId: 0,
                label: 'room',
              );
            }

            detections.add(detection);
          }
        }
      } catch (e) {
        continue;
      }
    }
  }

  static void _processYOLOv8Format(
      dynamic output,
      int numFeatures,
      int numDetections,
      double originalWidth,
      double originalHeight,
      double weight,
      List<Detection> detections,
      double confThreshold,
      [dynamic maskOutput,
      bool hasSegmentation = false]) {
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

          final left = (centerX - width / 2) * originalWidth / inputSize;
          final top = (centerY - height / 2) * originalHeight / inputSize;
          final detWidth = width * originalWidth / inputSize;
          final detHeight = height * originalHeight / inputSize;

          if (detWidth <= 0 || detHeight <= 0) continue;

          Detection detection;

          if (hasSegmentation && maskOutput != null) {
            final mask = _extractAndResizeMask(
                maskOutput, i, originalWidth.toInt(), originalHeight.toInt());
            detection = SegmentationDetection(
              left: left,
              top: top,
              width: detWidth,
              height: detHeight,
              confidence: confidence,
              classId: 0,
              label: 'room',
              mask: mask,
            );
          } else {
            detection = Detection(
              left: left,
              top: top,
              width: detWidth,
              height: detHeight,
              confidence: confidence,
              classId: 0,
              label: 'room',
            );
          }

          detections.add(detection);
        }
      } catch (e) {
        continue;
      }
    }
  }

  List<dynamic> _createOutputBuffers(Interpreter interpreter) {
    List<dynamic> outputs = [];
    final outputTensors = interpreter.getOutputTensors();

    for (int i = 0; i < outputTensors.length; i++) {
      final outputTensor = outputTensors[i];
      final shape = outputTensor.shape;

      logMessage(
          'Creating buffer for output $i with shape: $shape, type: ${outputTensor.type}');

      if (shape.length == 3) {
        outputs.add(List.generate(shape[0],
            (b) => List.generate(shape[1], (d) => List.filled(shape[2], 0.0))));
      } else if (shape.length == 4) {
        outputs.add(List.generate(
            shape[0],
            (b) => List.generate(
                shape[1],
                (h) => List.generate(
                    shape[2], (w) => List.filled(shape[3], 0.0)))));
      } else {
        throw Exception('Unsupported output shape: $shape for output $i');
      }
    }

    logMessage('Created ${outputs.length} output buffers');
    return outputs;
  }

  static List<List<double>> _extractAndResizeMask(dynamic maskOutput,
      int detectionIndex, int targetWidth, int targetHeight) {
    try {
      List<List<double>> rawMask = [];

      if (maskOutput is List && maskOutput.length > 0) {
        final batch = maskOutput[0];

        if (batch is List && batch.length > detectionIndex) {
          final maskData = batch[detectionIndex];
          if (maskData is List) {
            for (var row in maskData) {
              if (row is List) {
                rawMask.add(row.map((e) => (e as num).toDouble()).toList());
              }
            }
          }
        }
      }

      if (rawMask.isEmpty) {
        return List.generate(
            targetHeight, (_) => List.filled(targetWidth, 0.0));
      }

      _normalizeMask(rawMask);
      return _resizeMaskBilinear(rawMask, targetWidth, targetHeight);
    } catch (e) {
      print('Error extracting mask: $e');
      return List.generate(targetHeight, (_) => List.filled(targetWidth, 0.0));
    }
  }

  static void _normalizeMask(List<List<double>> mask) {
    if (mask.isEmpty || mask[0].isEmpty) return;

    double minVal = double.infinity;
    double maxVal = double.negativeInfinity;

    for (var row in mask) {
      for (var val in row) {
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }

    final range = maxVal - minVal;
    if (range > 0) {
      for (int i = 0; i < mask.length; i++) {
        for (int j = 0; j < mask[i].length; j++) {
          mask[i][j] = (mask[i][j] - minVal) / range;
        }
      }
    }
  }

  static List<List<double>> _resizeMaskBilinear(
      List<List<double>> mask, int newWidth, int newHeight) {
    if (mask.isEmpty || mask[0].isEmpty) {
      return List.generate(newHeight, (_) => List.filled(newWidth, 0.0));
    }

    final oldHeight = mask.length;
    final oldWidth = mask[0].length;

    final resizedMask =
        List.generate(newHeight, (_) => List.filled(newWidth, 0.0));

    final scaleX = oldWidth / newWidth;
    final scaleY = oldHeight / newHeight;

    for (int y = 0; y < newHeight; y++) {
      for (int x = 0; x < newWidth; x++) {
        final srcX = x * scaleX;
        final srcY = y * scaleY;

        final x1 = srcX.floor();
        final y1 = srcY.floor();
        final x2 = (x1 + 1).clamp(0, oldWidth - 1);
        final y2 = (y1 + 1).clamp(0, oldHeight - 1);

        final fx = srcX - x1;
        final fy = srcY - y1;

        final val1 = mask[y1][x1] * (1 - fx) + mask[y1][x2] * fx;
        final val2 = mask[y2][x1] * (1 - fx) + mask[y2][x2] * fx;

        resizedMask[y][x] = val1 * (1 - fy) + val2 * fy;
      }
    }

    return resizedMask;
  }

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

        if (group[0] is SegmentationDetection) {
          List<List<double>> avgMask = [];
          for (var detection in group) {
            if (detection is SegmentationDetection &&
                detection.mask.isNotEmpty) {
              avgMask = detection.mask;
              break;
            }
          }

          averaged.add(SegmentationDetection(
            left: avgLeft,
            top: avgTop,
            width: avgWidth,
            height: avgHeight,
            confidence: avgConfidence,
            classId: group[0].classId,
            label: group[0].label,
            mask: avgMask,
          ));
        } else {
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
    }

    return averaged;
  }

  List<Detection> _applyNMS(List<Detection> detections) {
    if (detections.isEmpty) return detections;

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

  // Helper method to calculate image variance for preprocessing validation
  static double _calculateImageVariance(img.Image image) {
    double mean = 0.0;
    int pixelCount = 0;

    // Sample from multiple regions to get variance
    final sampleSize = 20;
    final regions = [
      {'x': 0, 'y': 0},
      {'x': image.width - sampleSize, 'y': 0},
      {'x': 0, 'y': image.height - sampleSize},
      {'x': image.width - sampleSize, 'y': image.height - sampleSize},
      {
        'x': (image.width - sampleSize) ~/ 2,
        'y': (image.height - sampleSize) ~/ 2
      },
    ];

    // Calculate mean
    for (final region in regions) {
      int startX = region['x']!.clamp(0, image.width - 1);
      int startY = region['y']!.clamp(0, image.height - 1);
      int endX = (startX + sampleSize).clamp(0, image.width);
      int endY = (startY + sampleSize).clamp(0, image.height);

      for (int y = startY; y < endY; y++) {
        for (int x = startX; x < endX; x++) {
          mean += image.getPixel(x, y).r;
          pixelCount++;
        }
      }
    }
    mean /= pixelCount;

    // Calculate variance
    double variance = 0.0;
    for (final region in regions) {
      int startX = region['x']!.clamp(0, image.width - 1);
      int startY = region['y']!.clamp(0, image.height - 1);
      int endX = (startX + sampleSize).clamp(0, image.width);
      int endY = (startY + sampleSize).clamp(0, image.height);

      for (int y = startY; y < endY; y++) {
        for (int x = startX; x < endX; x++) {
          final pixelValue = image.getPixel(x, y).r;
          variance += (pixelValue - mean) * (pixelValue - mean);
        }
      }
    }
    variance /= pixelCount;

    return variance / (255.0 * 255.0); // Normalize to 0-1 range
  }
}
