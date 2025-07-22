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
    // Calculate scale factors
    double scaleX = size.width / imageSize.width;
    double scaleY = size.height / imageSize.height;

    // Define colors for different confidence levels
    final List<Color> confidenceColors = [
      Colors.red, // High confidence (80%+)
      Colors.orange, // Medium-high confidence (60-80%)
      Colors.yellow, // Medium confidence (40-60%)
      Colors.cyan, // Low-medium confidence (20-40%)
      Colors.purple, // Low confidence (<20%)
    ];

    final textPainter = TextPainter(
      textDirection: TextDirection.ltr,
    );

    for (int i = 0; i < detections.length; i++) {
      final detection = detections[i];

      // Scale bounding box coordinates
      double left = detection.left * scaleX;
      double top = detection.top * scaleY;
      double width = detection.width * scaleX;
      double height = detection.height * scaleY;

      // Choose color based on confidence level
      Color boxColor;
      if (detection.confidence >= 0.8) {
        boxColor = confidenceColors[0]; // Red for high confidence
      } else if (detection.confidence >= 0.6) {
        boxColor = confidenceColors[1]; // Orange
      } else if (detection.confidence >= 0.4) {
        boxColor = confidenceColors[2]; // Yellow
      } else if (detection.confidence >= 0.2) {
        boxColor = confidenceColors[3]; // Cyan
      } else {
        boxColor = confidenceColors[4]; // Purple
      }

      // Create paint for the bounding box outline
      final paint = Paint()
        ..color = boxColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = 4.0;

      // Create paint for the semi-transparent fill
      final fillPaint = Paint()
        ..color = boxColor.withOpacity(0.1)
        ..style = PaintingStyle.fill;

      // Create paint for the highlighted border
      final highlightPaint = Paint()
        ..color = Colors.white
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;

      final rect = Rect.fromLTWH(left, top, width, height);

      // Draw segmentation mask if available (behind the bounding box)
      if (detection is SegmentationDetection && detection.mask.isNotEmpty) {
        final maskPaint = Paint()
          ..color = boxColor.withOpacity(0.3)
          ..style = PaintingStyle.fill;
        _drawSegmentationMask(
            canvas, detection.mask, left, top, width, height, maskPaint);
      }

      // Draw semi-transparent fill
      canvas.drawRect(rect, fillPaint);

      // Draw white highlight border (inner)
      canvas.drawRect(rect, highlightPaint);

      // Draw main colored border (outer)
      canvas.drawRect(rect, paint);

      // Add corner markers for better visibility
      _drawCornerMarkers(canvas, rect, boxColor);

      // Draw detection number badge
      _drawDetectionBadge(canvas, left, top, i + 1, boxColor);

      // Draw label with confidence
      String label =
          '${detection.label} ${(detection.confidence * 100).toStringAsFixed(1)}%';
      if (detection is SegmentationDetection) {
        label += ' (seg)';
      }

      // Enhanced label styling with better contrast
      textPainter.text = TextSpan(
        text: label,
        style: TextStyle(
          color: Colors.white,
          fontSize: 14,
          fontWeight: FontWeight.bold,
          shadows: [
            Shadow(
              offset: const Offset(1, 1),
              color: Colors.black.withOpacity(0.8),
              blurRadius: 2,
            ),
          ],
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

      // Draw label background with rounded corners
      final labelRect = Rect.fromLTWH(
        labelX - 4,
        labelY - 2,
        textPainter.width + 8,
        textPainter.height + 4,
      );

      final labelBgPaint = Paint()
        ..color = boxColor.withOpacity(0.9)
        ..style = PaintingStyle.fill;

      final labelBorderPaint = Paint()
        ..color = Colors.white
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.0;

      // Draw rounded rectangle for label background
      final labelRRect =
          RRect.fromRectAndRadius(labelRect, const Radius.circular(4));
      canvas.drawRRect(labelRRect, labelBgPaint);
      canvas.drawRRect(labelRRect, labelBorderPaint);

      textPainter.paint(canvas, Offset(labelX, labelY));
    }
  }

  /// Draw corner markers for enhanced visibility
  void _drawCornerMarkers(Canvas canvas, Rect rect, Color color) {
    final cornerPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 6.0
      ..strokeCap = StrokeCap.round;

    const double cornerLength = 20.0;

    // Top-left corner
    canvas.drawLine(
      Offset(rect.left, rect.top),
      Offset(rect.left + cornerLength, rect.top),
      cornerPaint,
    );
    canvas.drawLine(
      Offset(rect.left, rect.top),
      Offset(rect.left, rect.top + cornerLength),
      cornerPaint,
    );

    // Top-right corner
    canvas.drawLine(
      Offset(rect.right, rect.top),
      Offset(rect.right - cornerLength, rect.top),
      cornerPaint,
    );
    canvas.drawLine(
      Offset(rect.right, rect.top),
      Offset(rect.right, rect.top + cornerLength),
      cornerPaint,
    );

    // Bottom-left corner
    canvas.drawLine(
      Offset(rect.left, rect.bottom),
      Offset(rect.left + cornerLength, rect.bottom),
      cornerPaint,
    );
    canvas.drawLine(
      Offset(rect.left, rect.bottom),
      Offset(rect.left, rect.bottom - cornerLength),
      cornerPaint,
    );

    // Bottom-right corner
    canvas.drawLine(
      Offset(rect.right, rect.bottom),
      Offset(rect.right - cornerLength, rect.bottom),
      cornerPaint,
    );
    canvas.drawLine(
      Offset(rect.right, rect.bottom),
      Offset(rect.right, rect.bottom - cornerLength),
      cornerPaint,
    );
  }

  /// Draw detection number badge
  void _drawDetectionBadge(
      Canvas canvas, double left, double top, int number, Color color) {
    const double badgeSize = 24.0;

    final badgePaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final badgeBorderPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    // Draw circle badge
    final center = Offset(left, top);
    canvas.drawCircle(center, badgeSize / 2, badgePaint);
    canvas.drawCircle(center, badgeSize / 2, badgeBorderPaint);

    // Draw number text
    final textPainter = TextPainter(
      text: TextSpan(
        text: number.toString(),
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    );

    textPainter.layout();
    final textOffset = Offset(
      left - textPainter.width / 2,
      top - textPainter.height / 2,
    );
    textPainter.paint(canvas, textOffset);
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
