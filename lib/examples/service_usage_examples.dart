import 'dart:io';
import 'package:floorplan_detection_app/interfaces/model_config.dart';
import 'package:floorplan_detection_app/services/room_detection_service_factory.dart';

/// Example of how to use the room detection services
class RoomDetectionExample {
  /// Example: Using Streamlined Service for a small image
  static Future<void> exampleStreamlinedDetection() async {
    print('=== Streamlined Detection Example ===');

    // Create streamlined service
    final service = RoomDetectionServiceFactory.createService(
      serviceType: DetectionServiceType.streamlined,
      modelConfigs: [
        ModelConfig(
            assetPath: 'assets/models/floorplan_v17.2.tflite', weight: 1.0),
        ModelConfig(
            assetPath: 'assets/models/floorplans-seg_v19.tflite', weight: 0.75),
      ],
      onLog: (message) => print('[STREAMLINED] $message'),
    );

    try {
      // Load models
      final loaded = await service.loadModels();
      if (!loaded) {
        print('Failed to load models');
        return;
      }

      // Process an image file
      final imageFile = File('path/to/your/image.jpg');
      if (await imageFile.exists()) {
        final detections = await service.processImageFile(imageFile);
        print('Found ${detections.length} rooms using streamlined detection');

        for (int i = 0; i < detections.length; i++) {
          final detection = detections[i];
          print(
              'Room ${i + 1}: ${detection.label} (${(detection.confidence * 100).toStringAsFixed(1)}%)');
        }
      }
    } catch (e) {
      print('Error: $e');
    } finally {
      service.dispose();
    }
  }

  /// Example: Using Tiled Service for a large image
  static Future<void> exampleTiledDetection() async {
    print('=== Tiled Detection Example ===');

    // Create tiled service
    final service = RoomDetectionServiceFactory.createService(
      serviceType: DetectionServiceType.tiled,
      modelConfigs: [
        ModelConfig(
            assetPath: 'assets/models/floorplan_v17.2.tflite', weight: 1.0),
        ModelConfig(
            assetPath: 'assets/models/floorplans-seg_v19.tflite', weight: 0.75),
      ],
      onLog: (message) => print('[TILED] $message'),
    );

    try {
      // Load models
      final loaded = await service.loadModels();
      if (!loaded) {
        print('Failed to load models');
        return;
      }

      // Process a large PDF file
      final pdfFile = File('path/to/your/floorplan.pdf');
      if (await pdfFile.exists()) {
        final detections = await service.processPDF(pdfFile, dpi: 150.0);
        print('Found ${detections.length} rooms using tiled detection');

        for (int i = 0; i < detections.length; i++) {
          final detection = detections[i];
          print(
              'Room ${i + 1}: ${detection.label} (${(detection.confidence * 100).toStringAsFixed(1)}%)');
          print(
              '  Location: (${detection.left.toStringAsFixed(1)}, ${detection.top.toStringAsFixed(1)})');
          print(
              '  Size: ${detection.width.toStringAsFixed(1)} x ${detection.height.toStringAsFixed(1)}');
        }
      }
    } catch (e) {
      print('Error: $e');
    } finally {
      service.dispose();
    }
  }

  /// Example: Choosing service automatically based on image size
  static Future<void> exampleAutoServiceSelection() async {
    print('=== Auto Service Selection Example ===');

    final imageFile = File('path/to/your/image.jpg');
    if (!await imageFile.exists()) {
      print('Image file not found');
      return;
    }

    // You would need to get image dimensions first (using image package)
    const imageWidth = 3000; // Example dimensions
    const imageHeight = 2000;

    // Get recommended service type
    final recommendedType =
        RoomDetectionServiceFactory.getRecommendedServiceType(
      imageWidth: imageWidth,
      imageHeight: imageHeight,
    );

    print(
        'Recommended service: ${RoomDetectionServiceFactory.getServiceDescription(recommendedType)}');

    // Create service based on recommendation
    final service = RoomDetectionServiceFactory.createService(
      serviceType: recommendedType,
      modelConfigs: [
        ModelConfig(
            assetPath: 'assets/models/floorplan_v17.2.tflite', weight: 1.0),
      ],
      onLog: (message) => print('[AUTO] $message'),
    );

    try {
      await service.loadModels();
      final detections = await service.processImageFile(imageFile);
      print('Auto-selected service found ${detections.length} rooms');
    } catch (e) {
      print('Error: $e');
    } finally {
      service.dispose();
    }
  }

  /// Example: Getting all available service types
  static void exampleListAllServices() {
    print('=== Available Detection Services ===');

    final allServices = RoomDetectionServiceFactory.getAllServiceTypes();

    for (final entry in allServices.entries) {
      print('${entry.key.name}: ${entry.value}');
    }
  }
}

/// Main function to run all examples
void main() async {
  print('Room Detection Service Examples\n');

  // List all available services
  RoomDetectionExample.exampleListAllServices();
  print('');

  // Note: Uncomment these to run actual detection examples
  // Make sure to update the file paths to actual files

  // await RoomDetectionExample.exampleStreamlinedDetection();
  // print('');

  // await RoomDetectionExample.exampleTiledDetection();
  // print('');

  // await RoomDetectionExample.exampleAutoServiceSelection();
}
