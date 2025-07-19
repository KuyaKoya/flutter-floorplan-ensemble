import 'dart:typed_data';
import 'package:floorplan_detection_app/interfaces/detection.dart';
import 'package:floorplan_detection_app/interfaces/model_config.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class RoomDetectionService {
  static const int inputSize = 640;
  static const double confidenceThreshold =
      0.3; // Lowered threshold for better detection
  static const double iouGroupingThreshold = 0.5;
  static const double nmsThreshold = 0.45;
  static const int roomClassId = 0; // Room class ID

  // List of room-related class names (you can expand this)
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

  RoomDetectionService({required this.modelConfigs});

  Future<bool> loadModels() async {
    try {
      for (var config in modelConfigs) {
        final interpreter = await Interpreter.fromAsset(config.assetPath);
        _interpreters.add(interpreter);
      }
      _isModelLoaded = true;
      print('Loaded ${_interpreters.length} models');
      return true;
    } catch (e) {
      print('Failed to load models: $e');
      _isModelLoaded = false;
      return false;
    }
  }

  bool get isModelLoaded => _isModelLoaded;

  Float32List preprocessImage(img.Image image) {
    final resized = img.copyResize(image, width: inputSize, height: inputSize);
    final bytes = Float32List(inputSize * inputSize * 3);
    int pixelIndex = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        bytes[pixelIndex++] = pixel.r / 255.0;
        bytes[pixelIndex++] = pixel.g / 255.0;
        bytes[pixelIndex++] = pixel.b / 255.0;
      }
    }

    return bytes;
  }

  Future<List<Detection>> runInference(img.Image image) async {
    if (!_isModelLoaded || _interpreters.isEmpty) {
      throw Exception('Models not loaded');
    }

    final input = preprocessImage(image);
    final inputData = [
      List.generate(inputSize, (h) {
        return List.generate(inputSize, (w) {
          return List.generate(3, (c) => input[(h * inputSize + w) * 3 + c]);
        });
      })
    ];

    List<Detection> allDetections = [];

    for (int i = 0; i < _interpreters.length; i++) {
      final interpreter = _interpreters[i];
      final weight = modelConfigs[i].weight;

      try {
        // Get output shape to determine model format
        final outputShape = interpreter.getOutputTensor(0).shape;
        print('Model $i output shape: $outputShape');

        // Dynamically create output buffer based on actual shape
        final output = _createOutputBuffer(outputShape);

        print('Running inference for model $i...');
        interpreter.run(inputData, output);
        print('Model $i inference completed successfully');

        final modelDetections = postprocessDetections(
          output,
          outputShape,
          image.width.toDouble(),
          image.height.toDouble(),
          weight,
        );

        print('Model $i produced ${modelDetections.length} detections');
        allDetections.addAll(modelDetections);

        // Yield control back to UI thread after each model to prevent freezing
        await Future.delayed(Duration.zero);
      } catch (e) {
        print('Model $i inference failed: $e');
        print('Model $i stack trace: ${StackTrace.current}');
      }
    }

    return applyNMS(averageOverlappingBoxes(allDetections));
  }

  /// Create output buffer based on the model's actual output shape
  dynamic _createOutputBuffer(List<int> shape) {
    if (shape.length == 3) {
      // YOLOv8 format: [batch, features, anchors]
      return List.generate(shape[0],
          (_) => List.generate(shape[1], (_) => List.filled(shape[2], 0.0)));
    } else if (shape.length == 3 && shape[1] > shape[2]) {
      // YOLOv5 format: [batch, anchors, features]
      return List.generate(shape[0],
          (_) => List.generate(shape[1], (_) => List.filled(shape[2], 0.0)));
    } else {
      // Fallback for other formats
      throw Exception('Unsupported output shape: $shape');
    }
  }

  List<Detection> postprocessDetections(
    dynamic output,
    List<int> outputShape,
    double originalWidth,
    double originalHeight,
    double confidenceWeight,
  ) {
    List<Detection> detections = [];
    double scaleX = originalWidth / inputSize;
    double scaleY = originalHeight / inputSize;

    // Determine if this is YOLOv8 or YOLOv5 format
    bool isYOLOv8 = outputShape.length == 3 && outputShape[1] > outputShape[2];
    print(
        'Processing ${isYOLOv8 ? "YOLOv8" : "YOLOv5"} format output: $outputShape');

    if (isYOLOv8) {
      // YOLOv8 format: [batch, features, anchors] -> [1, 37, 8400]
      // Features: [x, y, w, h, conf, class0, class1, ..., classN]
      final anchors = outputShape[2]; // 8400
      final features = outputShape[1]; // 37 (4 coords + 1 conf + 32 classes)

      for (int i = 0; i < anchors; i++) {
        // Extract coordinates and confidence
        double cx = output[0][0][i]; // center x
        double cy = output[0][1][i]; // center y
        double w = output[0][2][i]; // width
        double h = output[0][3][i]; // height
        double objectness = output[0][4][i]; // objectness score

        // Find best class (check multiple room-related classes)
        double maxClassScore = 0.0;
        int bestClassId = 0;
        String bestClassName = 'Room';

        for (int c = 5; c < features; c++) {
          double classScore = output[0][c][i];
          if (classScore > maxClassScore) {
            maxClassScore = classScore;
            bestClassId = c - 5; // Subtract 5 to get actual class index
            // Use class name if available, otherwise default to 'Room'
            if (bestClassId < roomClassNames.length) {
              bestClassName = roomClassNames[bestClassId];
            } else {
              bestClassName = 'Room_$bestClassId';
            }
          }
        }

        // Calculate final confidence (objectness * class_score)
        double confidence = objectness * maxClassScore * confidenceWeight;

        // Accept any room-related class or class 0
        bool isRoomClass =
            bestClassId < roomClassNames.length || bestClassId == roomClassId;

        if (confidence >= confidenceThreshold && isRoomClass) {
          // Convert from center coordinates to top-left coordinates
          double left = (cx - w / 2) * scaleX;
          double top = (cy - h / 2) * scaleY;
          double width = w * scaleX;
          double height = h * scaleY;

          detections.add(Detection(
            left: left,
            top: top,
            width: width,
            height: height,
            confidence: confidence,
            classId: bestClassId,
            label: bestClassName,
          ));
        }
      }
    } else {
      // YOLOv5 format: [batch, anchors, features] -> [1, 25200, 6]
      // Features: [x, y, w, h, conf, class]
      for (var det in output[0]) {
        double confidence = det[4] * confidenceWeight;
        int classId = det[5].round();

        if (confidence >= confidenceThreshold && classId == roomClassId) {
          double cx = det[0] * scaleX;
          double cy = det[1] * scaleY;
          double w = det[2] * scaleX;
          double h = det[3] * scaleY;
          double left = cx - w / 2;
          double top = cy - h / 2;

          detections.add(Detection(
            left: left,
            top: top,
            width: w,
            height: h,
            confidence: confidence,
            classId: classId,
            label: 'Room',
          ));
        }
      }
    }

    print(
        'Extracted ${detections.length} detections from ${isYOLOv8 ? "YOLOv8" : "YOLOv5"} output');
    return detections;
  }

  List<Detection> averageOverlappingBoxes(List<Detection> detections) {
    List<Detection> finalDetections = [];
    List<bool> merged = List.filled(detections.length, false);

    for (int i = 0; i < detections.length; i++) {
      if (merged[i]) continue;

      List<Detection> group = [detections[i]];
      merged[i] = true;

      for (int j = i + 1; j < detections.length; j++) {
        if (merged[j]) continue;
        if (calculateIoU(detections[i], detections[j]) >=
            iouGroupingThreshold) {
          group.add(detections[j]);
          merged[j] = true;
        }
      }

      // Weighted box averaging
      double totalWeight = group.fold(0.0, (sum, d) => sum + d.confidence);
      double avgLeft = 0,
          avgTop = 0,
          avgWidth = 0,
          avgHeight = 0,
          avgConfidence = 0;

      for (var d in group) {
        avgLeft += d.left * d.confidence;
        avgTop += d.top * d.confidence;
        avgWidth += d.width * d.confidence;
        avgHeight += d.height * d.confidence;
        avgConfidence += d.confidence;
      }

      finalDetections.add(Detection(
        left: avgLeft / totalWeight,
        top: avgTop / totalWeight,
        width: avgWidth / totalWeight,
        height: avgHeight / totalWeight,
        confidence: avgConfidence / group.length,
        classId: roomClassId,
        label: 'Room',
      ));
    }

    return finalDetections;
  }

  List<Detection> applyNMS(List<Detection> detections) {
    if (detections.isEmpty) return [];
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));

    List<Detection> result = [];
    List<bool> suppressed = List.filled(detections.length, false);

    for (int i = 0; i < detections.length; i++) {
      if (suppressed[i]) continue;
      result.add(detections[i]);

      for (int j = i + 1; j < detections.length; j++) {
        if (suppressed[j]) continue;
        if (calculateIoU(detections[i], detections[j]) > nmsThreshold) {
          suppressed[j] = true;
        }
      }
    }

    return result;
  }

  double calculateIoU(Detection a, Detection b) {
    double left = a.left > b.left ? a.left : b.left;
    double top = a.top > b.top ? a.top : b.top;
    double right = (a.left + a.width) < (b.left + b.width)
        ? (a.left + a.width)
        : (b.left + b.width);
    double bottom = (a.top + a.height) < (b.top + b.height)
        ? (a.top + a.height)
        : (b.top + b.height);

    if (right <= left || bottom <= top) return 0.0;

    double interArea = (right - left) * (bottom - top);
    double unionArea = a.width * a.height + b.width * b.height - interArea;

    return interArea / unionArea;
  }

  void dispose() {
    for (final interpreter in _interpreters) {
      interpreter.close();
    }
    _interpreters.clear();
    _isModelLoaded = false;
  }
}
