# Floor Plan Room Detection App

A Flutter application that uses on-device YOLO (You Only Look Once) machine learning to detect rooms in floor plan images using TensorFlow Lite. Now supports PDF import and large image tiling for processing architectural drawings.

## Features

- **On-device AI inference** using TensorFlow Lite (no internet required)
- **YOLO room detection** for architectural floor plans
- **PDF import and conversion** to raster images for processing
- **Large image tiling** with overlap handling for processing big floor plans
- **Multi-model ensemble** support with weighted confidence averaging
- **Advanced post-processing** with box averaging and NMS across tiles
- **Image input** from camera, gallery, PDF files, or sample assets
- **Real-time visualization** of detected rooms with bounding boxes
- **Cross-platform** support (iOS, Android, Web, Desktop)

## Technology Stack

- **Flutter** - Cross-platform UI framework
- **TensorFlow Lite** - On-device machine learning inference
- **YOLO** - Object detection model architecture
- **Image processing** - Built-in Dart image manipulation
- **PDF processing** - PDF to image conversion
- **File picking** - Platform-native file selection

## New Architecture Overview

### Core Services

1. **`RoomDetectionService`** - Base YOLO inference with multi-model support
2. **`PdfProcessingService`** - PDF import and image tiling utilities
3. **`TiledRoomDetectionService`** - Enhanced detection with tiling pipeline
4. **`FloorplanProcessingDemo`** - Complete workflow demonstrations

### Processing Pipeline

```
PDF File → PDF to Image → Create Tiles → Process Each Tile → 
Transform Coordinates → Global Post-processing → Final Detections
```

### Key Improvements

- **Tiling System**: Splits large images into 640×640 overlapping tiles
- **Coordinate Transformation**: Maps tile-local detections to global coordinates
- **Overlap Handling**: Prevents double-counting in overlapping regions
- **Ensemble Detection**: Combines multiple models with weighted confidence
- **Advanced NMS**: Global non-maximum suppression across all tiles

## Setup Instructions

### 1. Prerequisites

- Flutter SDK (3.6.1 or later)
- Dart SDK
- Android Studio / Xcode for mobile development
- A trained YOLOv5 TensorFlow Lite model for room detection

### 2. Install Dependencies

```bash
flutter pub get
```

### 3. Add YOLO Model

Place your trained YOLOv5 TensorFlow Lite model in the assets folder:

```
assets/models/yolov5.tflite
```

**Model Requirements:**

- Input size: 640x640 pixels
- Output format: [x, y, w, h, confidence, class_id]
- Class index for "room": 0
- TensorFlow Lite format (.tflite)

### 4. Add Sample Images (Optional)

Add test floor plan images to:

```
assets/images/sample_floorplan.png
```

### 5. Model Conversion (If needed)

If you have a PyTorch YOLOv5 model, convert it to TensorFlow Lite:

```bash
pip install ultralytics
yolo export model=yolov5s.pt format=tflite imgsz=640
```

## Usage

1. **Launch the app** on your device/emulator
2. **Wait for model loading** (shown in status message)
3. **Select an image** using one of these options:
   - 📷 **Camera**: Take a photo of a floor plan
   - 🖼️ **Gallery**: Choose from existing photos
   - 📋 **Sample**: Use bundled test image
4. **View results**: Detected rooms appear with red bounding boxes and confidence scores

## App Architecture

### Core Components

- **`RoomDetectionService`** - Handles TensorFlow Lite model loading and inference
- **`FloorPlanDetectionScreen`** - Main UI screen with image selection and results
- **`DetectionOverlayPainter`** - Custom painter for drawing bounding boxes
- **`Detection`** - Data class for detection results

### Key Features

#### Image Preprocessing

- Resize to 640x640 pixels
- RGB normalization to [0, 1] range
- Tensor format conversion for YOLO input

#### YOLO Inference

- On-device TensorFlow Lite execution
- No network connectivity required
- Real-time processing

#### Post-processing

- Confidence threshold filtering (50%)
- Non-Maximum Suppression (NMS) for duplicate removal
- Coordinate scaling to original image dimensions

#### Visualization

- Custom painter overlay on original images
- Bounding box rendering with labels
- Confidence percentage display

## Customization

### Adjust Detection Parameters

Edit `lib/services/room_detection_service.dart`:

```dart
static const double confidenceThreshold = 0.5;  // Minimum confidence
static const double nmsThreshold = 0.45;        // NMS threshold
static const int roomClassId = 0;               // Target class ID
```

### Model Path

Update the model path in `RoomDetectionService`:

```dart
static const String modelPath = 'assets/models/your_model.tflite';
```

### UI Customization

Modify colors, layouts, and styling in:

- `lib/screens/floorplan_detection_screen.dart`
- `lib/widgets/detection_overlay.dart`

## Troubleshooting

### Common Issues

1. **Model not loading**
   - Ensure `yolov5.tflite` exists in `assets/models/`
   - Check file path in `pubspec.yaml` assets section
   - Verify model is compatible TensorFlow Lite format

2. **No detections found**
   - Check confidence threshold (try lowering to 0.1)
   - Verify model is trained for room detection
   - Ensure input image contains clear room boundaries

3. **Performance issues**
   - Use smaller input size (e.g., 320x320)
   - Optimize model with quantization
   - Test on physical device vs emulator

### Debug Information

The app prints detection results to console:

```
Detected 3 rooms:
Detection(label: Room, confidence: 0.85, box: [120.0, 50.0, 200.0, 150.0])
Detection(label: Room, confidence: 0.72, box: [350.0, 80.0, 180.0, 120.0])
...
```

## Model Training (Advanced)

To train your own room detection model:

1. **Collect floor plan images** with room annotations
2. **Label bounding boxes** using tools like LabelImg or Roboflow
3. **Train YOLOv5** with your dataset
4. **Export to TensorFlow Lite** format
5. **Test and iterate** on model performance

## Dependencies

- `tflite_flutter: ^0.9.1` - TensorFlow Lite inference
- `image: ^3.3.0` - Image processing and manipulation
- `image_picker: ^1.0.7` - Camera and gallery access
- `path_provider: ^2.0.15` - File system access

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

**Note**: This app requires a pre-trained YOLO model for room detection. The model file is not included and must be provided separately.
