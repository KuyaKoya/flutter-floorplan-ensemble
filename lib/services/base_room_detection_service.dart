import 'dart:io';
import 'package:floorplan_detection_app/interfaces/detection.dart';
import 'package:floorplan_detection_app/interfaces/model_config.dart';

/// Abstract base class for room detection services
abstract class BaseRoomDetectionService {
  final List<ModelConfig> modelConfigs;
  final Function(String)? onLog;

  BaseRoomDetectionService({
    required this.modelConfigs,
    this.onLog,
  });

  /// Load all models asynchronously
  Future<bool> loadModels();

  /// Check if models are loaded
  bool get isModelLoaded;

  /// Process a PDF file
  Future<List<Detection>> processPDF(File pdfFile, {double dpi = 150.0});

  /// Process an image file
  Future<List<Detection>> processImageFile(File imageFile);

  /// Dispose resources
  void dispose();

  /// Helper method for logging
  void logMessage(String message) {
    if (onLog != null) {
      onLog!(message);
    }
    print(message);
  }
}
