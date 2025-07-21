import 'package:floorplan_detection_app/interfaces/model_config.dart';
import 'package:floorplan_detection_app/services/base_room_detection_service.dart';
import 'package:floorplan_detection_app/services/streamlined_room_detection_service.dart';
import 'package:floorplan_detection_app/services/tiled_room_detection_service.dart';

/// Enum for selecting detection service type
enum DetectionServiceType {
  streamlined,
  tiled,
}

/// Factory class for creating room detection services
class RoomDetectionServiceFactory {
  /// Create a room detection service based on the specified type
  static BaseRoomDetectionService createService({
    required DetectionServiceType serviceType,
    required List<ModelConfig> modelConfigs,
    Function(String)? onLog,
  }) {
    switch (serviceType) {
      case DetectionServiceType.streamlined:
        return StreamlinedRoomDetectionService(
          modelConfigs: modelConfigs,
          onLog: onLog,
        );
      case DetectionServiceType.tiled:
        return TiledRoomDetectionService(
          modelConfigs: modelConfigs,
          onLog: onLog,
        );
    }
  }

  /// Get a human-readable description of the service type
  static String getServiceDescription(DetectionServiceType serviceType) {
    switch (serviceType) {
      case DetectionServiceType.streamlined:
        return 'Streamlined Detection (Full Image Resize)';
      case DetectionServiceType.tiled:
        return 'Tiled Detection (Large Image Tiling)';
    }
  }

  /// Get recommended service type based on image characteristics
  static DetectionServiceType getRecommendedServiceType({
    required int imageWidth,
    required int imageHeight,
    int maxStreamlinedSize = 2000,
  }) {
    // Recommend tiled processing for large images
    if (imageWidth > maxStreamlinedSize || imageHeight > maxStreamlinedSize) {
      return DetectionServiceType.tiled;
    }

    // Recommend streamlined for smaller images
    return DetectionServiceType.streamlined;
  }

  /// Get all available service types with descriptions
  static Map<DetectionServiceType, String> getAllServiceTypes() {
    return {
      for (final type in DetectionServiceType.values)
        type: getServiceDescription(type),
    };
  }
}
