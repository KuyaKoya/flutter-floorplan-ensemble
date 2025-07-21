import 'package:flutter/material.dart';

import '../interfaces/detection.dart';

class DetectionOverlayPainter extends CustomPainter {
  final List<Detection> detections;
  final Size imageSize;

  DetectionOverlayPainter({
    required this.detections,
    required this.imageSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final maskPaint = Paint()
      ..color = Colors.blue.withOpacity(0.3)
      ..style = PaintingStyle.fill;

    final textPainter = TextPainter(
      textDirection: TextDirection.ltr,
    );

    // Calculate scale factors
    double scaleX = size.width / imageSize.width;
    double scaleY = size.height / imageSize.height;

    for (final detection in detections) {
      // Scale bounding box coordinates
      double left = detection.left * scaleX;
      double top = detection.top * scaleY;
      double width = detection.width * scaleX;
      double height = detection.height * scaleY;

      // Draw segmentation mask if available
      if (detection is SegmentationDetection && detection.mask.isNotEmpty) {
        _drawSegmentationMask(
            canvas, detection.mask, left, top, width, height, maskPaint);
      }

      // Draw bounding box
      final rect = Rect.fromLTWH(left, top, width, height);
      canvas.drawRect(rect, paint);

      // Draw label with confidence
      String label =
          '${detection.label} ${(detection.confidence * 100).toStringAsFixed(1)}%';
      if (detection is SegmentationDetection) {
        label += ' (segmented)';
      }

      textPainter.text = TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.red,
          fontSize: 16,
          fontWeight: FontWeight.bold,
          backgroundColor: Colors.white,
        ),
      );
      textPainter.text = TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.red,
          fontSize: 16,
          fontWeight: FontWeight.bold,
          backgroundColor: Colors.white,
        ),
      );

      textPainter.layout();

      // Position label above the bounding box
      double labelX = left;
      double labelY = top - textPainter.height - 5;

      // Ensure label stays within canvas bounds
      if (labelY < 0) labelY = top + 5;
      if (labelX + textPainter.width > size.width) {
        labelX = size.width - textPainter.width;
      }

      // Draw label background
      final labelRect = Rect.fromLTWH(
        labelX - 2,
        labelY - 2,
        textPainter.width + 4,
        textPainter.height + 4,
      );

      canvas.drawRect(
        labelRect,
        Paint()..color = Colors.white.withOpacity(0.8),
      );

      textPainter.paint(canvas, Offset(labelX, labelY));
    }
  }

  /// Draw segmentation mask overlay
  void _drawSegmentationMask(Canvas canvas, List<List<double>> mask,
      double left, double top, double width, double height, Paint maskPaint) {
    if (mask.isEmpty || mask[0].isEmpty) return;

    final maskHeight = mask.length;
    final maskWidth = mask[0].length;

    final cellWidth = width / maskWidth;
    final cellHeight = height / maskHeight;

    for (int y = 0; y < maskHeight; y++) {
      for (int x = 0; x < maskWidth; x++) {
        final maskValue = mask[y][x];

        // Only draw pixels above threshold (0.5)
        if (maskValue > 0.5) {
          final cellLeft = left + x * cellWidth;
          final cellTop = top + y * cellHeight;

          final cellRect =
              Rect.fromLTWH(cellLeft, cellTop, cellWidth, cellHeight);

          // Vary opacity based on mask confidence
          final paint = Paint()
            ..color = Colors.blue.withOpacity(maskValue * 0.5)
            ..style = PaintingStyle.fill;

          canvas.drawRect(cellRect, paint);
        }
      }
    }
  }

  @override
  bool shouldRepaint(DetectionOverlayPainter oldDelegate) {
    return detections != oldDelegate.detections ||
        imageSize != oldDelegate.imageSize;
  }
}
