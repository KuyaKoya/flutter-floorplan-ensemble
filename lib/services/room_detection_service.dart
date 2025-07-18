import 'dart:typed_data';
import 'package:floorplan_detection_app/interfaces/detection.dart';
import 'package:floorplan_detection_app/interfaces/model_config.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class RoomDetectionService {
  static const int inputSize = 640;
  static const double confidenceThreshold = 0.5;
  static const double iouGroupingThreshold = 0.5;
  static const double nmsThreshold = 0.45;
  static const int roomClassId = 0;

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

      final output = List.generate(
          1, (_) => List.generate(25200, (_) => List.filled(6, 0.0)));

      try {
        interpreter.run(inputData, output);
        allDetections.addAll(postprocessDetections(
          output,
          image.width.toDouble(),
          image.height.toDouble(),
          weight,
        ));
      } catch (e) {
        print('Model $i inference failed: $e');
      }
    }

    return applyNMS(averageOverlappingBoxes(allDetections));
  }

  List<Detection> postprocessDetections(
    List<List<List<double>>> output,
    double originalWidth,
    double originalHeight,
    double confidenceWeight,
  ) {
    List<Detection> detections = [];
    double scaleX = originalWidth / inputSize;
    double scaleY = originalHeight / inputSize;

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
