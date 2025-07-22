# Floor Plan Room Detection App

A Flutter application that uses on-device YOLO machine learning to detect and segment rooms in floor plan images using TensorFlow Lite. Features advanced multi-model ensemble detection with PDF processing and intelligent image tiling for architectural drawings.

## ✨ Features

- **🤖 On-device AI inference** using TensorFlow Lite (no internet required)
- **🏠 Advanced room detection** with multiple YOLO models and segmentation
- **📄 PDF import and processing** with automatic conversion to images
- **🧩 Intelligent image tiling** with overlap handling for large floor plans
- **🎯 Multi-model ensemble** support with weighted confidence averaging
- **⚡ Dual processing modes** - Streamlined for small images, Tiled for large ones
- **📱 Multiple input sources** - Camera, gallery, PDF files, or sample assets
- **🎨 Real-time visualization** with bounding boxes and confidence scores
- **🌐 Cross-platform support** (iOS, Android, Web, Desktop)
- **📊 Live processing logs** with detailed operation tracking

## 🏗️ Architecture Overview

### Core Services

The app uses a factory pattern with multiple specialized detection services:

1. **`StreamlinedRoomDetectionService`** - Direct model inference for small-to-medium images
2. **`TiledRoomDetectionService`** - Advanced tiling pipeline for large images  
3. **`PdfProcessingService`** - PDF to image conversion and preprocessing
4. **`BaseRoomDetectionService`** - Abstract interface for all detection services

### Available Models

The app includes three pre-trained models in `assets/models/`:

- **`floorplan_v17.2.tflite`** - Primary detection model (weight: 1.0)
- **`floorplans-seg_v19.tflite`** - Segmentation model (weight: 0.75)  
- **`floorplan_v12.tflite`** - Alternative detection model

### Processing Pipeline

**Streamlined Mode (Small Images):**

```text
Image Input → Resize to 640×640 → Multi-Model Inference → 
Ensemble Averaging → NMS → Coordinate Scaling → Results
```

**Tiled Mode (Large Images):**

```text
Large Image → Create Overlapping Tiles → Process Each Tile → 
Transform Coordinates → Global Post-processing → Final Detections
```

### Key Capabilities

- **🔄 Automatic service selection** based on image size
- **🎛️ Manual service switching** via UI toggle
- **📝 Comprehensive logging** with real-time status updates
- **⚖️ Weighted model ensemble** for improved accuracy
- **🎯 Advanced NMS** to eliminate duplicate detections
- **📐 Precise coordinate transformation** from tile-space to image-space

## 🛠️ Technology Stack

- **Flutter 3.6.1+** - Cross-platform UI framework with Material 3 design
- **TensorFlow Lite 0.11.0** - On-device machine learning inference
- **YOLO Architecture** - Object detection and segmentation models
- **Dart Image 4.1.7** - Advanced image processing and manipulation
- **PDF Processing** - Built-in PDF to image conversion (pdf: 3.10.7)
- **File System Access** - Camera, gallery, and PDF file picking
- **Printing 5.12.0** - PDF generation and export capabilities

## 🚀 Getting Started

### Prerequisites

- **Flutter SDK 3.6.1+**
- **Dart SDK** (included with Flutter)
- **Android Studio** or **Xcode** for mobile development
- **VS Code** with Flutter extensions (recommended)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd floorplan_detection_app
   ```

2. **Install dependencies:**

   ```bash
   flutter pub get
   ```

3. **Verify models are present:**

   ```bash
   ls assets/models/
   # Should show: floorplan_v12.tflite, floorplan_v17.2.tflite, floorplans-seg_v19.tflite
   ```

4. **Run the app:**

   ```bash
   flutter run
   # Or use the VS Code task: "Flutter: Run Debug"
   ```

## 📱 How to Use

### Basic Workflow

1. **🚀 Launch the app** and wait for model initialization
2. **📊 Monitor the status** - Models will load automatically (shows progress)
3. **⚙️ Choose processing mode:**
   - **Streamlined**: Best for small-medium images (faster)
   - **Tiled**: Best for large architectural drawings (more accurate)
4. **📸 Select input source:**
   - **📷 Camera**: Take a photo of a floor plan
   - **🖼️ Gallery**: Choose from existing images
   - **📄 PDF**: Import and process PDF floor plans
   - **📋 Sample**: Use bundled test images
5. **🔍 View results**: Detected rooms appear with colored bounding boxes
6. **📝 Check logs**: Detailed processing information in the bottom panel

### Service Selection Guide

**When to use Streamlined Mode:**

- Images smaller than 2000×2000 pixels
- Quick processing needed
- Simple floor plans with clear room boundaries

**When to use Tiled Mode:**

- Large architectural drawings (>2000×2000 pixels)
- Complex floor plans with many small rooms
- Maximum detection accuracy required
- PDF imports of detailed blueprints

## 🏛️ App Architecture

### Core Components

#### Services Layer

- **`BaseRoomDetectionService`** - Abstract interface for all detection services
- **`StreamlinedRoomDetectionService`** - Direct inference for standard-sized images
- **`TiledRoomDetectionService`** - Advanced tiling pipeline for large images
- **`PdfProcessingService`** - PDF to image conversion and preprocessing
- **`RoomDetectionServiceFactory`** - Factory pattern for service creation

#### UI Layer

- **`FloorPlanDetectionScreen`** - Main application screen with full functionality
- **`DetectionOverlayPainter`** - Custom painter for rendering detection results
- **Material 3 Design** - Modern, adaptive UI components

#### Data Layer

- **`Detection`** - Core data class for detection results
- **`SegmentationDetection`** - Extended detection with mask data
- **`ModelConfig`** - Configuration for model loading and weighting

### Key Features Deep Dive

#### 🔄 Multi-Model Ensemble

The app automatically loads and combines multiple models:

```dart
ModelConfig(assetPath: 'assets/models/floorplan_v17.2.tflite', weight: 1.0),
ModelConfig(assetPath: 'assets/models/floorplans-seg_v19.tflite', weight: 0.75),
```

- **Weighted averaging** of confidence scores
- **Redundancy** for improved accuracy
- **Fallback support** if one model fails

#### 🧩 Intelligent Tiling System

For large images, the app automatically:

1. **Splits into 640×640 overlapping tiles**
2. **Processes each tile independently**
3. **Transforms coordinates** back to global space
4. **Applies global NMS** to remove duplicates
5. **Merges results** across all tiles

#### 📊 Real-time Logging

Comprehensive logging system provides:

- **Model loading progress**
- **Processing pipeline status**
- **Detection counts and confidence scores**
- **Performance metrics and timing**
- **Error handling and debugging info**

#### ⚙️ Adaptive Processing

The app automatically recommends processing mode based on:

- **Image dimensions** (threshold: 2000×2000 pixels)
- **Available memory**
- **Processing complexity**

### Image Processing Pipeline

#### Preprocessing

- **Format conversion** to RGB if needed
- **Resize** to model input size (640×640)
- **Normalization** to [0, 1] range
- **Tensor preparation** for TensorFlow Lite

#### Inference

- **Multi-model execution** in parallel
- **Confidence scoring** with weighted averaging
- **Class filtering** (room detection only)
- **Memory-efficient** batch processing

#### Post-processing

- **Confidence thresholding** (default: 50%)
- **Non-Maximum Suppression** (NMS threshold: 45%)
- **Coordinate scaling** to original image dimensions
- **Bounding box refinement** and validation

## ⚙️ Configuration & Customization

### Detection Parameters

You can adjust detection sensitivity in the service files:

**Confidence Threshold** (in `streamlined_room_detection_service.dart`):

```dart
static const double confidenceThreshold = 0.5;  // Minimum confidence (50%)
```

**NMS Threshold** (for duplicate removal):

```dart
static const double nmsThreshold = 0.45;        // Overlap threshold (45%)
```

**Model Weights** (in `floorplan_detection_screen.dart`):

```dart
ModelConfig(assetPath: 'assets/models/floorplan_v17.2.tflite', weight: 1.0),
ModelConfig(assetPath: 'assets/models/floorplans-seg_v19.tflite', weight: 0.75),
```

### Service Selection

**Manual Override** (in the UI):

- Toggle between "Streamlined" and "Tiled" modes
- App provides recommendations based on image size

**Automatic Selection Threshold**:

```dart
static DetectionServiceType getRecommendedServiceType({
  required int imageWidth,
  required int imageHeight,
  int maxStreamlinedSize = 2000,  // Configurable threshold
})
```

### UI Customization

**Theme Colors** (in `main.dart`):

```dart
theme: ThemeData(
  colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
  useMaterial3: true,
),
```

**Detection Overlay** (in `detection_overlay.dart`):

- Bounding box colors and styles
- Confidence score display format
- Label positioning and fonts

## 🐛 Troubleshooting

### Common Issues

#### 1. Models Not Loading

**Symptoms:** "Failed to load models" error message

**Solutions:**

- Verify all `.tflite` files exist in `assets/models/`
- Check `pubspec.yaml` includes the assets folder
- Ensure models are not corrupted (download again if needed)
- Check console logs for specific TensorFlow Lite errors

#### 2. No Detections Found

**Symptoms:** "No rooms detected" message with clear floor plans

**Solutions:**

- Lower confidence threshold temporarily: `confidenceThreshold = 0.1`
- Try switching between Streamlined and Tiled modes
- Ensure image shows clear room boundaries and walls
- Check if floor plan is similar to training data (architectural drawings)

#### 3. Poor Detection Accuracy

**Symptoms:** Missing rooms or false positives

**Solutions:**

- Use **Tiled mode** for large, detailed floor plans
- Adjust model weights in the ensemble
- Try different combinations of available models
- Ensure good image quality and resolution

#### 4. App Performance Issues

**Symptoms:** Slow processing or memory errors

**Solutions:**

- Use **Streamlined mode** for smaller images
- Close other apps to free memory
- Test on physical device vs emulator
- Reduce image size before processing

#### 5. PDF Processing Fails

**Symptoms:** Errors when importing PDF files

**Solutions:**

- Ensure PDF contains vector graphics or high-resolution images
- Try converting PDF to image externally first
- Check PDF file is not password-protected
- Verify file size is reasonable (< 50MB recommended)

### Debug Information

The app provides comprehensive logging in the bottom panel:

```text
=== Starting Model Initialization ===
Loading Streamlined Detection (Full Image Resize) models...
✓ Model loaded: assets/models/floorplan_v17.2.tflite (Weight: 1.0)
✓ Model loaded: assets/models/floorplans-seg_v19.tflite (Weight: 0.75)
✓ All models loaded successfully
Models loaded. Select an image or PDF to analyze.
```

**Detection Results Example:**

```text
Detected 3 rooms:
Detection(label: Room, confidence: 0.85, box: [120.0, 50.0, 200.0, 150.0])
Detection(label: Room, confidence: 0.72, box: [350.0, 80.0, 180.0, 120.0])
Detection(label: Room, confidence: 0.68, box: [50.0, 200.0, 150.0, 100.0])
```

### Performance Benchmarks

**Typical Processing Times:**

- **Small images** (< 1000×1000): 1-3 seconds
- **Medium images** (1000-2000×2000): 3-8 seconds  
- **Large images** (> 2000×2000): 10-30 seconds (tiled)
- **PDF conversion**: 2-5 seconds per page

**Memory Usage:**

- **Base app**: ~100-200 MB
- **Models loaded**: +150-300 MB
- **Image processing**: +50-200 MB (depending on size)

## 📚 Dependencies

### Core Dependencies

```yaml
dependencies:
  flutter: sdk: flutter
  tflite_flutter: ^0.11.0      # TensorFlow Lite inference engine
  image: ^4.1.7                # Image processing and manipulation
  image_picker: ^1.0.7         # Camera and gallery access
  path_provider: ^2.1.2        # File system path access
  file_picker: ^8.0.0+1        # PDF and file selection
  printing: ^5.12.0            # PDF generation and export
  pdf: ^3.10.7                 # PDF processing utilities
```

### Development Dependencies

```yaml
dev_dependencies:
  flutter_test: sdk: flutter
  flutter_lints: ^5.0.0        # Dart/Flutter linting rules
```

### Model Requirements

**Included Models:**

- **Input size:** 640×640 pixels
- **Output format:** Bounding boxes + confidence scores
- **Classes:** Room detection (class_id: 0)
- **Format:** TensorFlow Lite (.tflite)
- **Architecture:** YOLO-based object detection/segmentation

## 🧪 Testing & Development

### Running Tests

```bash
# Run all tests
flutter test

# Run tests with coverage
flutter test --coverage

# Run integration tests (if available)
flutter test integration_test/
```

### Building for Production

```bash
# Android APK
flutter build apk --release

# Android App Bundle
flutter build appbundle --release

# iOS (requires Xcode and Apple Developer account)
flutter build ios --release

# Web
flutter build web --release

# Desktop (macOS)
flutter build macos --release
```

### Development Workflow

1. **Make changes** to service or UI code
2. **Hot reload** for UI changes (`r` in terminal)
3. **Hot restart** for logic changes (`R` in terminal)
4. **Test with different images** and service modes
5. **Check logs** for debugging information
6. **Run tests** before committing

## 🔬 Advanced Usage

### Adding Custom Models

To add your own trained models:

1. **Place model file** in `assets/models/`
2. **Update model configuration:**

```dart
ModelConfig(
  assetPath: 'assets/models/your_custom_model.tflite', 
  weight: 1.0
),
```

3. **Rebuild and test** the app

### API Usage Examples

**Programmatic Detection:**

```dart
// Create service
final service = RoomDetectionServiceFactory.createService(
  serviceType: DetectionServiceType.streamlined,
  modelConfigs: [
    ModelConfig(assetPath: 'assets/models/floorplan_v17.2.tflite', weight: 1.0),
  ],
);

// Load models
await service.loadModels();

// Process image
final detections = await service.processImageFile(imageFile);

// Use results
for (final detection in detections) {
  print('Room found: ${detection.confidence}');
}
```

### Performance Optimization

**For Better Speed:**

- Use Streamlined mode for smaller images
- Reduce model ensemble size
- Lower confidence threshold to reduce processing

**For Better Accuracy:**

- Use Tiled mode for large images
- Include multiple models in ensemble
- Increase overlap in tiling system

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch:** `git checkout -b feature/your-feature-name`
4. **Make your changes** and test thoroughly
5. **Follow coding standards** (use `flutter analyze`)
6. **Submit a pull request** with detailed description

### Contribution Guidelines

- **Code Style:** Follow Dart/Flutter conventions
- **Testing:** Add tests for new features
- **Documentation:** Update README for significant changes
- **Commits:** Use clear, descriptive commit messages
- **Issues:** Report bugs with detailed reproduction steps

### Areas for Contribution

- **🧠 Model improvements** (training, optimization)
- **🎨 UI/UX enhancements** (Material 3, accessibility)
- **⚡ Performance optimizations** (memory, speed)
- **🧪 Testing coverage** (unit, integration, widget tests)
- **📖 Documentation** (API docs, tutorials, examples)
- **🌐 Platform support** (Web, Desktop improvements)

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Flutter Team** for the excellent cross-platform framework
- **TensorFlow Team** for TensorFlow Lite mobile inference
- **YOLO** researchers for object detection architecture
- **Open source community** for various packages and tools

---

**⚠️ Note:** This app includes pre-trained models for demonstration. For production use, consider training models on your specific floor plan dataset for optimal accuracy.

**🚀 Ready to detect rooms?** Launch the app and start analyzing your floor plans!
