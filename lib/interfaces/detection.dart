/// Data class for detection results
class Detection {
  final double left;
  final double top;
  final double width;
  final double height;
  final double confidence;
  final int classId;
  final String label;

  Detection({
    required this.left,
    required this.top,
    required this.width,
    required this.height,
    required this.confidence,
    required this.classId,
    required this.label,
  });

  @override
  String toString() {
    return 'Detection(label: $label, confidence: ${confidence.toStringAsFixed(2)}, '
        'box: [${left.toStringAsFixed(1)}, ${top.toStringAsFixed(1)}, '
        '${width.toStringAsFixed(1)}, ${height.toStringAsFixed(1)}])';
  }
}

/// Extended detection class for segmentation models with masks
class SegmentationDetection extends Detection {
  final List<List<double>> mask;

  SegmentationDetection({
    required super.left,
    required super.top,
    required super.width,
    required super.height,
    required super.confidence,
    required super.classId,
    required super.label,
    required this.mask,
  });

  @override
  String toString() {
    return 'SegmentationDetection(label: $label, confidence: ${confidence.toStringAsFixed(2)}, '
        'box: [${left.toStringAsFixed(1)}, ${top.toStringAsFixed(1)}, '
        '${width.toStringAsFixed(1)}, ${height.toStringAsFixed(1)}], '
        'mask: ${mask.length}x${mask.isNotEmpty ? mask[0].length : 0})';
  }
}
