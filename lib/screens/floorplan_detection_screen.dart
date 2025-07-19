import 'dart:io';
import 'dart:async';
import 'package:floorplan_detection_app/interfaces/detection.dart';
import 'package:floorplan_detection_app/interfaces/model_config.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import '../services/streamlined_room_detection_service.dart';
import '../services/pdf_processing_service.dart';
import '../widgets/detection_overlay.dart';

class FloorPlanDetectionScreen extends StatefulWidget {
  const FloorPlanDetectionScreen({super.key});

  @override
  State<FloorPlanDetectionScreen> createState() =>
      _FloorPlanDetectionScreenState();
}

class _FloorPlanDetectionScreenState extends State<FloorPlanDetectionScreen> {
  late final StreamlinedRoomDetectionService _detectionService;
  final ImagePicker _imagePicker = ImagePicker();
  final ScrollController _logScrollController = ScrollController();

  File? _selectedImage;
  File? _selectedPdf;
  img.Image? _processedImage;
  List<Detection> _detections = [];
  bool _isLoading = false;
  bool _isModelLoaded = false;
  String _statusMessage = 'Initializing...';
  final List<String> _processLogs = [];

  @override
  void initState() {
    super.initState();

    // Initialize the streamlined detection service with logging callback
    _detectionService = StreamlinedRoomDetectionService(
      modelConfigs: [
        ModelConfig(
            assetPath: 'assets/models/floorplan_v17.2.tflite', weight: 1.0),
        ModelConfig(
            assetPath: 'assets/models/floorplan_v12.tflite', weight: 0.75),
      ],
      onLog: _addLog,
    );

    _addLog('Flutter Floor Plan Detection App Started');
    _addLog('Initializing streamlined room detection service (no tiling)...');
    _initializeModel();
  }

  @override
  void dispose() {
    _addLog('Disposing resources and closing app...');
    _detectionService.dispose();
    super.dispose();
  }

  Future<void> _initializeModel() async {
    _addLog('=== Starting Model Initialization ===');
    setState(() {
      _isLoading = true;
      _statusMessage = 'Loading YOLO models...';
    });

    try {
      _addLog('Loading StreamlinedRoomDetectionService models...');
      bool success = await _detectionService.loadModels();

      if (success) {
        _addLog('✓ All models loaded successfully');
        setState(() {
          _isModelLoaded = success;
          _statusMessage =
              'Models loaded. Select an image or PDF to analyze (full image processing).';
          _isLoading = false;
        });
      } else {
        _addLog('✗ Failed to load models');
        setState(() {
          _isModelLoaded = success;
          _statusMessage =
              'Failed to load models. Please ensure model files are in assets/models/';
          _isLoading = false;
        });
      }
    } catch (e) {
      _addLog('✗ Error during model loading: $e');
      setState(() {
        _isModelLoaded = false;
        _statusMessage = 'Error loading models: $e';
        _isLoading = false;
      });
    }
    _addLog('=== Model Initialization Complete ===');
  }

  Future<void> _pickImageFromGallery() async {
    if (!_isModelLoaded) {
      _addLog('✗ Cannot pick image: Model not loaded yet');
      _showErrorSnackBar('Model not loaded yet');
      return;
    }

    try {
      _addLog('Opening image gallery picker...');
      final XFile? image = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 100,
      );

      if (image != null) {
        _addLog('✓ Image selected from gallery: ${image.path}');
        await _processImage(File(image.path));
      } else {
        _addLog('Image selection cancelled by user');
      }
    } catch (e) {
      _addLog('✗ Error picking image from gallery: $e');
      _showErrorSnackBar('Error picking image: $e');
    }
  }

  Future<void> _pickImageFromCamera() async {
    if (!_isModelLoaded) {
      _addLog('✗ Cannot take photo: Model not loaded yet');
      _showErrorSnackBar('Model not loaded yet');
      return;
    }

    try {
      _addLog('Opening camera...');
      final XFile? image = await _imagePicker.pickImage(
        source: ImageSource.camera,
        imageQuality: 100,
      );

      if (image != null) {
        _addLog('✓ Photo taken successfully: ${image.path}');
        await _processImage(File(image.path));
      } else {
        _addLog('Photo capture cancelled by user');
      }
    } catch (e) {
      _addLog('✗ Error taking photo: $e');
      _showErrorSnackBar('Error taking photo: $e');
    }
  }

  Future<void> _loadSampleImage() async {
    if (!_isModelLoaded) {
      _addLog('✗ Cannot load sample: Model not loaded yet');
      _showErrorSnackBar('Model not loaded yet');
      return;
    }

    try {
      _addLog('Loading sample image from assets...');
      // Try to load a sample image from assets
      final ByteData data =
          await rootBundle.load('assets/images/sample_floorplan.png');
      final Uint8List bytes = data.buffer.asUint8List();
      _addLog('Sample image loaded: ${bytes.length} bytes');

      // Create a temporary file
      final tempDir = Directory.systemTemp;
      final tempFile = File('${tempDir.path}/sample_floorplan.png');
      await tempFile.writeAsBytes(bytes);
      _addLog('✓ Sample image saved to temp file: ${tempFile.path}');

      await _processImage(tempFile);
    } catch (e) {
      _addLog('✗ Sample image not found: $e');
      _showErrorSnackBar(
          'Sample image not found. Please add a sample image to assets/images/');
    }
  }

  Future<void> _pickPdfFile() async {
    if (!_isModelLoaded) {
      _addLog('✗ Cannot pick PDF: Model not loaded yet');
      _showErrorSnackBar('Model not loaded yet');
      return;
    }

    try {
      _addLog('Opening PDF file picker...');
      final File? pdfFile = await PdfProcessingService.pickPdfFile();

      if (pdfFile != null) {
        _addLog('✓ PDF file selected: ${pdfFile.path}');
        await _processPdf(pdfFile);
      } else {
        _addLog('PDF selection cancelled by user');
      }
    } catch (e) {
      _addLog('✗ Error picking PDF: $e');
      _showErrorSnackBar('Error picking PDF: $e');
    }
  }

  Future<void> _processImage(File imageFile) async {
    _addLog('=== Starting Image Processing ===');
    _addLog('Input file: ${imageFile.path}');

    setState(() {
      _isLoading = true;
      _statusMessage = 'Processing image (full image, no tiling)...';
      _selectedImage = imageFile;
      _selectedPdf = null;
      _detections = [];
    });

    try {
      // Use the streamlined service to process the full image
      _addLog('Processing full image with streamlined detection service...');
      _addLog('Image will be resized to 640x640 regardless of original size');

      final List<Detection> detections =
          await _detectionService.processImageFile(imageFile);

      _addLog('✓ Detection complete: ${detections.length} rooms found');

      if (detections.isEmpty) {
        _addLog('⚠️ No rooms detected. This could be due to:');
        _addLog('  - Model confidence threshold too low (currently ${0.1})');
        _addLog('  - Model designed for single \'room\' class only');
        _addLog('  - Image doesn\'t contain recognizable room features');
        _addLog('  - Model not trained for this type of floorplan');
        _addLog('  - Model output processing errors (check for null values)');
        _addLog('  - Image resolution/quality issues');
      }

      // Load the processed image for display
      final imageBytes = await imageFile.readAsBytes();
      final image = img.decodeImage(imageBytes);

      setState(() {
        _detections = detections;
        _processedImage = image;
        _statusMessage = 'Found ${detections.length} room(s)';
        _isLoading = false;
      });

      // Log detection results
      _addLog('=== Detection Results ===');
      for (int i = 0; i < detections.length; i++) {
        final detection = detections[i];
        _addLog(
            'Room ${i + 1}: ${detection.label} (${(detection.confidence * 100).toStringAsFixed(1)}% confidence)');
      }
      _addLog('=== Image Processing Complete ===');
    } catch (e) {
      _addLog('✗ Error processing image: $e');
      setState(() {
        _statusMessage = 'Error processing image: $e';
        _isLoading = false;
      });
      _showErrorSnackBar('Error processing image: $e');
    }
  }

  Future<void> _processPdf(File pdfFile) async {
    _addLog('=== Starting PDF Processing ===');
    _addLog('Input PDF file: ${pdfFile.path}');

    setState(() {
      _isLoading = true;
      _statusMessage = 'Processing PDF (full page, no tiling)...';
      _selectedPdf = pdfFile;
      _selectedImage = null;
      _detections = [];
    });

    try {
      // Get file size info
      final fileSize = await pdfFile.length();
      _addLog('PDF file size: ${(fileSize / 1024).toStringAsFixed(1)} KB');

      // Use the streamlined service to process the PDF
      _addLog('Processing full PDF page with streamlined detection service...');
      _addLog(
          'PDF will be rendered and resized to 640x640 regardless of original size');

      final List<Detection> detections =
          await _detectionService.processPDF(pdfFile);

      _addLog('✓ PDF processing complete: ${detections.length} rooms found');

      if (detections.isEmpty) {
        _addLog('⚠️ No rooms detected in PDF. This could be due to:');
        _addLog('  - PDF rasterization quality too low');
        _addLog('  - Model confidence threshold too low (currently ${0.1})');
        _addLog('  - Model designed for single \'room\' class only');
        _addLog(
            '  - PDF contains architectural drawings not suitable for room detection');
        _addLog(
            '  - Model output processing errors (check console for null values)');
        _addLog('  - PDF complexity too high for current model');
      }

      // Get the processed image for display
      final img.Image? processedImage =
          await PdfProcessingService.pdfToImage(pdfFile);

      if (processedImage != null) {
        _addLog(
            '✓ PDF converted to image: ${processedImage.width}x${processedImage.height} pixels');
      }

      setState(() {
        _detections = detections;
        _processedImage = processedImage;
        _statusMessage = 'Found ${detections.length} room(s) in PDF';
        _isLoading = false;
      });

      // Log detection results
      _addLog('=== PDF Detection Results ===');
      for (int i = 0; i < detections.length; i++) {
        final detection = detections[i];
        _addLog(
            'Room ${i + 1}: ${detection.label} (${(detection.confidence * 100).toStringAsFixed(1)}% confidence)');
      }
      _addLog('=== PDF Processing Complete ===');
    } catch (e) {
      _addLog('✗ Error processing PDF: $e');
      setState(() {
        _statusMessage = 'Error processing PDF: $e';
        _isLoading = false;
      });
      _showErrorSnackBar('Error processing PDF: $e');
    }
  }

  /// Add a log message to both console and UI
  void _addLog(String message) {
    final timestamp = DateTime.now().toString().substring(11, 19);
    final logMessage = '[$timestamp] $message';

    // Console logging
    print(logMessage);

    // UI logging
    setState(() {
      _processLogs.add(logMessage);
      // Keep only last 100 logs to prevent memory issues
      if (_processLogs.length > 100) {
        _processLogs.removeAt(0);
      }
    });

    // Auto-scroll to bottom after a short delay
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_logScrollController.hasClients) {
        _logScrollController.animateTo(
          _logScrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  /// Clear all logs
  void _clearLogs() {
    setState(() {
      _processLogs.clear();
    });
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
      body: SingleChildScrollView(
        child: Column(
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
                  Wrap(
                    spacing: 8.0,
                    runSpacing: 8.0,
                    alignment: WrapAlignment.center,
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
                        onPressed: _isLoading ? null : _pickPdfFile,
                        icon: const Icon(Icons.picture_as_pdf),
                        label: const Text('PDF'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.red.shade100,
                          foregroundColor: Colors.red.shade700,
                        ),
                      ),
                      ElevatedButton.icon(
                        onPressed: _isLoading ? null : _loadSampleImage,
                        icon: const Icon(Icons.image),
                        label: const Text('Sample'),
                      ),
                    ],
                  ),

                  // Live logs section (always visible)
                  const SizedBox(height: 16),
                  Container(
                    height: 300, // Fixed height for logs
                    margin: const EdgeInsets.symmetric(horizontal: 8.0),
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.blue.shade300),
                      borderRadius: BorderRadius.circular(8.0),
                      color: Colors.blue.shade50,
                    ),
                    child: Column(
                      children: [
                        // Logs header
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(8.0),
                          decoration: BoxDecoration(
                            color: Colors.blue.shade200,
                            borderRadius: const BorderRadius.only(
                              topLeft: Radius.circular(8.0),
                              topRight: Radius.circular(8.0),
                            ),
                          ),
                          child: Row(
                            children: [
                              const Icon(Icons.terminal, size: 16),
                              const SizedBox(width: 8),
                              const Text(
                                'Live Process Logs',
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 14,
                                ),
                              ),
                              const Spacer(),
                              if (_processLogs.isNotEmpty)
                                TextButton.icon(
                                  onPressed: _clearLogs,
                                  icon: const Icon(Icons.clear, size: 16),
                                  label: const Text('Clear'),
                                  style: TextButton.styleFrom(
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 8.0),
                                    minimumSize: const Size(0, 32),
                                  ),
                                ),
                            ],
                          ),
                        ),
                        // Logs content
                        Expanded(
                          child: _processLogs.isEmpty
                              ? const Center(
                                  child: Text(
                                    'Ready to process. Select an image or PDF to see live logs.',
                                    style: TextStyle(
                                      color: Colors.grey,
                                      fontStyle: FontStyle.italic,
                                    ),
                                  ),
                                )
                              : ListView.builder(
                                  controller: _logScrollController,
                                  padding: const EdgeInsets.all(8.0),
                                  itemCount: _processLogs.length,
                                  itemBuilder: (context, index) {
                                    final log = _processLogs[index];
                                    final isError = log.contains('✗');
                                    final isSuccess = log.contains('✓');
                                    final isHeader = log.contains('===');
                                    final isWarning = log.contains('⚠️');

                                    return Padding(
                                      padding:
                                          const EdgeInsets.only(bottom: 2.0),
                                      child: Text(
                                        log,
                                        style: TextStyle(
                                          fontFamily: 'monospace',
                                          fontSize: 12,
                                          color: isError
                                              ? Colors.red.shade700
                                              : isSuccess
                                                  ? Colors.green.shade700
                                                  : isWarning
                                                      ? Colors.orange.shade700
                                                      : isHeader
                                                          ? Colors.blue.shade700
                                                          : Colors.black87,
                                          fontWeight: isHeader
                                              ? FontWeight.bold
                                              : FontWeight.normal,
                                        ),
                                      ),
                                    );
                                  },
                                ),
                        ),
                      ],
                    ),
                  ),
                  if (_isLoading)
                    const Padding(
                      padding: EdgeInsets.only(top: 16.0),
                      child: CircularProgressIndicator(),
                    ),
                ],
              ),
            ),

            // Image/PDF display with detections
            Container(
              height: 400, // Fixed height for image display
              width: double.infinity,
              margin: const EdgeInsets.all(16.0),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(8.0),
              ),
              child: (_selectedImage != null || _selectedPdf != null)
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(8.0),
                      child: Stack(
                        children: [
                          // Display image (either from file or processed from PDF)
                          if (_selectedImage != null)
                            Image.file(
                              _selectedImage!,
                              fit: BoxFit.contain,
                              width: double.infinity,
                              height: double.infinity,
                            )
                          else if (_processedImage != null)
                            Image.memory(
                              img.encodePng(_processedImage!),
                              fit: BoxFit.contain,
                              width: double.infinity,
                              height: double.infinity,
                            ),

                          // Show PDF info overlay if it's a PDF
                          if (_selectedPdf != null)
                            Positioned(
                              top: 8,
                              left: 8,
                              child: Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 8.0,
                                  vertical: 4.0,
                                ),
                                decoration: BoxDecoration(
                                  color: Colors.red.shade700,
                                  borderRadius: BorderRadius.circular(4.0),
                                ),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    const Icon(
                                      Icons.picture_as_pdf,
                                      color: Colors.white,
                                      size: 16,
                                    ),
                                    const SizedBox(width: 4),
                                    Text(
                                      'PDF',
                                      style: const TextStyle(
                                        color: Colors.white,
                                        fontSize: 12,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
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
                    )
                  : const Center(
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
                            'Select an image or PDF floorplan to analyze',
                            style: TextStyle(
                              fontSize: 18,
                              color: Colors.grey,
                            ),
                            textAlign: TextAlign.center,
                          ),
                          SizedBox(height: 8),
                          Text(
                            'Supports images from gallery/camera and PDF files',
                            style: TextStyle(
                              fontSize: 14,
                              color: Colors.grey,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ],
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

            // Add some bottom padding to ensure content doesn't get cut off
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }
}
