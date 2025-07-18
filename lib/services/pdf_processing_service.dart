import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:printing/printing.dart';
import 'package:image/image.dart' as img;

/// Service for handling PDF import and image tiling operations
class PdfProcessingService {
  static const int tileSize = 640;
  static const int tileOverlap = 64; // 10% overlap
  static const int effectiveTileSize = tileSize - tileOverlap;

  /// Pick a PDF file from the file system
  static Future<File?> pickPdfFile() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf'],
        allowMultiple: false,
      );

      if (result != null && result.files.single.path != null) {
        return File(result.files.single.path!);
      }
      return null;
    } catch (e) {
      print('Error picking PDF file: $e');
      return null;
    }
  }

  /// Convert the first page of a PDF to an img.Image object
  static Future<img.Image?> pdfToImage(File pdfFile,
      {double dpi = 150.0}) async {
    try {
      print('Loading PDF document: ${pdfFile.path}');

      // Read PDF file as bytes
      final pdfBytes = await pdfFile.readAsBytes();

      // Use the printing package to rasterize the PDF
      // The printing package provides raster functionality via a stream
      final rasterStream = Printing.raster(
        pdfBytes,
        pages: [0], // First page only
        dpi: dpi,
      );

      // Get the first page from the stream
      final pageData = await rasterStream.first;

      print('PDF rasterized successfully');

      // Convert the rasterized page to img.Image
      // PdfRaster provides width, height, and pixels
      final image = img.Image.fromBytes(
        width: pageData.width,
        height: pageData.height,
        bytes: pageData.pixels.buffer,
      );

      print('PDF converted to image: ${image.width}x${image.height}');
      return image;
    } catch (e) {
      print('Error converting PDF to image: $e');
      return null;
    }
  }

  /// Split a large image into overlapping tiles for processing
  static List<ImageTile> createImageTiles(img.Image image) {
    final tiles = <ImageTile>[];
    final imageWidth = image.width;
    final imageHeight = image.height;

    print('Creating tiles for image: ${imageWidth}x${imageHeight}');
    print('Tile size: ${tileSize}x${tileSize} with ${tileOverlap}px overlap');

    int tileIndex = 0;

    // Iterate through the image in tile-sized steps
    for (int y = 0; y < imageHeight; y += effectiveTileSize) {
      for (int x = 0; x < imageWidth; x += effectiveTileSize) {
        // Calculate actual tile dimensions (handle edge cases)
        final tileWidth =
            (x + tileSize > imageWidth) ? imageWidth - x : tileSize;
        final tileHeight =
            (y + tileSize > imageHeight) ? imageHeight - y : tileSize;

        // Extract the tile from the source image
        final tileImage = img.copyCrop(
          image,
          x: x,
          y: y,
          width: tileWidth,
          height: tileHeight,
        );

        // Pad the tile if it's smaller than expected (edge tiles)
        final paddedTile = _padTileIfNeeded(tileImage, tileSize, tileSize);

        final tile = ImageTile(
          image: paddedTile,
          offsetX: x,
          offsetY: y,
          originalWidth: tileWidth,
          originalHeight: tileHeight,
          index: tileIndex++,
        );

        tiles.add(tile);

        print(
            'Created tile ${tile.index}: ${tileWidth}x${tileHeight} at (${x}, ${y})');
      }
    }

    print('Created ${tiles.length} tiles total');
    return tiles;
  }

  /// Pad a tile image to the required size if it's smaller (for edge tiles)
  static img.Image _padTileIfNeeded(
      img.Image tile, int targetWidth, int targetHeight) {
    if (tile.width == targetWidth && tile.height == targetHeight) {
      return tile;
    }

    // Create a new image with the target size and fill with white background
    final paddedImage = img.fill(
      img.Image(width: targetWidth, height: targetHeight),
      color: img.ColorRgb8(255, 255, 255), // White background
    );

    // Copy the original tile onto the padded image
    img.compositeImage(
      paddedImage,
      tile,
      dstX: 0,
      dstY: 0,
    );

    print(
        'Padded tile from ${tile.width}x${tile.height} to ${targetWidth}x${targetHeight}');
    return paddedImage;
  }

  /// Calculate effective processing area considering overlap
  static ProcessingBounds calculateProcessingBounds(
      ImageTile tile, int imageWidth, int imageHeight) {
    // Calculate the actual processing area within the tile to avoid double-counting overlaps
    final halfOverlap = tileOverlap ~/ 2;

    int processLeft = 0;
    int processTop = 0;
    int processWidth = tile.originalWidth;
    int processHeight = tile.originalHeight;

    // Adjust for overlaps (except for edge tiles)
    if (tile.offsetX > 0) {
      processLeft = halfOverlap;
      processWidth -= halfOverlap;
    }

    if (tile.offsetY > 0) {
      processTop = halfOverlap;
      processHeight -= halfOverlap;
    }

    if (tile.offsetX + tile.originalWidth < imageWidth) {
      processWidth -= halfOverlap;
    }

    if (tile.offsetY + tile.originalHeight < imageHeight) {
      processHeight -= halfOverlap;
    }

    return ProcessingBounds(
      left: processLeft,
      top: processTop,
      width: processWidth,
      height: processHeight,
    );
  }
}

/// Represents a tile extracted from a larger image
class ImageTile {
  final img.Image image;
  final int offsetX; // X offset in the original image
  final int offsetY; // Y offset in the original image
  final int originalWidth; // Width before padding
  final int originalHeight; // Height before padding
  final int index; // Tile index for debugging

  const ImageTile({
    required this.image,
    required this.offsetX,
    required this.offsetY,
    required this.originalWidth,
    required this.originalHeight,
    required this.index,
  });

  @override
  String toString() {
    return 'ImageTile(index: $index, offset: ($offsetX, $offsetY), '
        'size: ${originalWidth}x$originalHeight, '
        'padded: ${image.width}x${image.height})';
  }
}

/// Represents the processing bounds within a tile to avoid overlap issues
class ProcessingBounds {
  final int left;
  final int top;
  final int width;
  final int height;

  const ProcessingBounds({
    required this.left,
    required this.top,
    required this.width,
    required this.height,
  });

  @override
  String toString() {
    return 'ProcessingBounds(left: $left, top: $top, width: $width, height: $height)';
  }
}
