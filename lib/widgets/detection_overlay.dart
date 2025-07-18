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

      // Draw bounding box
      final rect = Rect.fromLTWH(left, top, width, height);
      canvas.drawRect(rect, paint);

      // Draw label with confidence
      final label =
          '${detection.label} ${(detection.confidence * 100).toStringAsFixed(1)}%';

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

  @override
  bool shouldRepaint(DetectionOverlayPainter oldDelegate) {
    return detections != oldDelegate.detections ||
        imageSize != oldDelegate.imageSize;
  }
}
