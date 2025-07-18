import 'dart:io';
import 'package:floorplan_detection_app/interfaces/detection.dart';
import 'package:floorplan_detection_app/interfaces/model_config.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import '../services/room_detection_service.dart';
import '../widgets/detection_overlay.dart';

class FloorPlanDetectionScreen extends StatefulWidget {
  const FloorPlanDetectionScreen({super.key});

  @override
  State<FloorPlanDetectionScreen> createState() =>
      _FloorPlanDetectionScreenState();
}

class _FloorPlanDetectionScreenState extends State<FloorPlanDetectionScreen> {
  final RoomDetectionService _detectionService = RoomDetectionService(
    modelConfigs: [
      ModelConfig(
          assetPath: 'assets/models/floorplan_v17.2.tflite', weight: 1.0),
      ModelConfig(
          assetPath: 'assets/models/floorplan_v12.tflite', weight: 0.75),
    ],
  );
  final ImagePicker _imagePicker = ImagePicker();

  File? _selectedImage;
  img.Image? _processedImage;
  List<Detection> _detections = [];
  bool _isLoading = false;
  bool _isModelLoaded = false;
  String _statusMessage = 'Initializing...';

  @override
  void initState() {
    super.initState();
    _initializeModel();
  }

  @override
  void dispose() {
    _detectionService.dispose();
    super.dispose();
  }

  Future<void> _initializeModel() async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Loading YOLO model...';
    });

    try {
      bool success = await _detectionService.loadModels();
      setState(() {
        _isModelLoaded = success;
        _statusMessage = success
            ? 'Model loaded successfully. Select an image to analyze.'
            : 'Failed to load model. Please ensure yolov5.tflite is in assets/models/';
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isModelLoaded = false;
        _statusMessage = 'Error loading model: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _pickImageFromGallery() async {
    if (!_isModelLoaded) {
      _showErrorSnackBar('Model not loaded yet');
      return;
    }

    try {
      final XFile? image = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 100,
      );

      if (image != null) {
        await _processImage(File(image.path));
      }
    } catch (e) {
      _showErrorSnackBar('Error picking image: $e');
    }
  }

  Future<void> _pickImageFromCamera() async {
    if (!_isModelLoaded) {
      _showErrorSnackBar('Model not loaded yet');
      return;
    }

    try {
      final XFile? image = await _imagePicker.pickImage(
        source: ImageSource.camera,
        imageQuality: 100,
      );

      if (image != null) {
        await _processImage(File(image.path));
      }
    } catch (e) {
      _showErrorSnackBar('Error taking photo: $e');
    }
  }

  Future<void> _loadSampleImage() async {
    if (!_isModelLoaded) {
      _showErrorSnackBar('Model not loaded yet');
      return;
    }

    try {
      // Try to load a sample image from assets
      final ByteData data =
          await rootBundle.load('assets/images/sample_floorplan.png');
      final Uint8List bytes = data.buffer.asUint8List();

      // Create a temporary file
      final tempDir = Directory.systemTemp;
      final tempFile = File('${tempDir.path}/sample_floorplan.png');
      await tempFile.writeAsBytes(bytes);

      await _processImage(tempFile);
    } catch (e) {
      _showErrorSnackBar(
          'Sample image not found. Please add a sample image to assets/images/');
    }
  }

  Future<void> _processImage(File imageFile) async {
    setState(() {
      _isLoading = true;
      _statusMessage = 'Processing image...';
      _selectedImage = imageFile;
      _detections = [];
    });

    try {
      // Load and decode image
      final Uint8List imageBytes = await imageFile.readAsBytes();
      final img.Image? image = img.decodeImage(imageBytes);

      if (image == null) {
        throw Exception('Failed to decode image');
      }

      setState(() {
        _processedImage = image;
        _statusMessage = 'Running room detection...';
      });

      // Run inference
      final List<Detection> detections =
          await _detectionService.runInference(image);

      setState(() {
        _detections = detections;
        _statusMessage = 'Found ${detections.length} room(s)';
        _isLoading = false;
      });

      // Print detection results
      print('Detected ${detections.length} rooms:');
      for (final detection in detections) {
        print(detection.toString());
      }
    } catch (e) {
      setState(() {
        _statusMessage = 'Error processing image: $e';
        _isLoading = false;
      });
      _showErrorSnackBar('Error processing image: $e');
    }
  }

  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Floor Plan Room Detection'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Column(
        children: [
          // Status and controls
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: [
                Text(
                  _statusMessage,
                  style: Theme.of(context).textTheme.bodyLarge,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    ElevatedButton.icon(
                      onPressed: _isLoading ? null : _pickImageFromGallery,
                      icon: const Icon(Icons.photo_library),
                      label: const Text('Gallery'),
                    ),
                    ElevatedButton.icon(
                      onPressed: _isLoading ? null : _pickImageFromCamera,
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Camera'),
                    ),
                    ElevatedButton.icon(
                      onPressed: _isLoading ? null : _loadSampleImage,
                      icon: const Icon(Icons.image),
                      label: const Text('Sample'),
                    ),
                  ],
                ),
                if (_isLoading)
                  const Padding(
                    padding: EdgeInsets.only(top: 16.0),
                    child: CircularProgressIndicator(),
                  ),
              ],
            ),
          ),

          // Image display with detections
          Expanded(
            child: _selectedImage != null
                ? Container(
                    width: double.infinity,
                    margin: const EdgeInsets.all(16.0),
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(8.0),
                      child: Stack(
                        children: [
                          // Original image
                          Image.file(
                            _selectedImage!,
                            fit: BoxFit.contain,
                            width: double.infinity,
                            height: double.infinity,
                          ),

                          // Detection overlay
                          if (_processedImage != null && _detections.isNotEmpty)
                            Positioned.fill(
                              child: CustomPaint(
                                painter: DetectionOverlayPainter(
                                  detections: _detections,
                                  imageSize: Size(
                                    _processedImage!.width.toDouble(),
                                    _processedImage!.height.toDouble(),
                                  ),
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                  )
                : Container(
                    margin: const EdgeInsets.all(16.0),
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey),
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                    child: const Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.image_outlined,
                            size: 64,
                            color: Colors.grey,
                          ),
                          SizedBox(height: 16),
                          Text(
                            'Select an image to analyze',
                            style: TextStyle(
                              fontSize: 18,
                              color: Colors.grey,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
          ),

          // Detection results summary
          if (_detections.isNotEmpty)
            Container(
              width: double.infinity,
              margin: const EdgeInsets.all(16.0),
              padding: const EdgeInsets.all(16.0),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                borderRadius: BorderRadius.circular(8.0),
                border: Border.all(color: Colors.blue.shade200),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Detection Results:',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: Colors.blue.shade800,
                        ),
                  ),
                  const SizedBox(height: 8),
                  ..._detections.map((detection) => Padding(
                        padding: const EdgeInsets.only(bottom: 4.0),
                        child: Text(
                          '• ${detection.label}: ${(detection.confidence * 100).toStringAsFixed(1)}% confidence',
                          style: TextStyle(color: Colors.blue.shade700),
                        ),
                      )),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
