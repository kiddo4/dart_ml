import 'dart:io';
import 'dart:math';
import 'package:image/image.dart' as img;

class Visualization {
  final int width;
  final int height;
  final int padding;
  final img.BitmapFont font;

  Visualization({this.width = 800, this.height = 600, this.padding = 40, required this.font});

  void drawLinePlot(List<double> xValues, List<double> yValues, String filePath, {String title = ''}) {
    _checkEqualLength(xValues, yValues);

    final image = img.Image(width: width, height: height);
    img.fill(image, color: img.ColorRgb8(255, 255, 255)); // White background

    final minMax = _getMinMax(xValues, yValues);
    final minX = minMax[0], maxX = minMax[1], minY = minMax[2], maxY = minMax[3];

    // Draw axes
    _drawAxes(image, minX, maxX, minY, maxY);

    // Draw data points and connect them with lines
    for (int i = 0; i < xValues.length; i++) {
      final x = _normalize(xValues[i], minX, maxX, width - 2 * padding) + padding;
      final y = height - (_normalize(yValues[i], minY, maxY, height - 2 * padding) + padding);

      // Draw point
      _drawCircle(image, x, y, 3, img.ColorRgb8(0, 0, 255));

      // Connect points with lines
      if (i > 0) {
        final prevX = _normalize(xValues[i - 1], minX, maxX, width - 2 * padding) + padding;
        final prevY = height - (_normalize(yValues[i - 1], minY, maxY, height - 2 * padding) + padding);
        _drawLine(image, prevX, prevY, x, y, img.ColorRgb8(0, 0, 255));
      }
    }

    _addTitleAndLabels(image, minX, maxX, minY, maxY, title);

    // Save the image to a file
    File(filePath).writeAsBytesSync(img.encodeJpg(image));
  }

  void drawScatterPlot(List<double> xValues, List<double> yValues, String filePath, {String title = ''}) {
    _checkEqualLength(xValues, yValues);

    final image = img.Image(width: width, height: height);
    img.fill(image, color: img.ColorRgb8(255, 255, 255)); // White background

    final minMax = _getMinMax(xValues, yValues);
    final minX = minMax[0], maxX = minMax[1], minY = minMax[2], maxY = minMax[3];

    // Draw axes
    _drawAxes(image, minX, maxX, minY, maxY);

    // Draw scatter plot points
    for (int i = 0; i < xValues.length; i++) {
      final x = _normalize(xValues[i], minX, maxX, width - 2 * padding) + padding;
      final y = height - (_normalize(yValues[i], minY, maxY, height - 2 * padding) + padding);
      _drawCircle(image, x, y, 4, img.ColorRgb8(255, 0, 0));
    }

    _addTitleAndLabels(image, minX, maxX, minY, maxY, title);

    // Save the image to a file
    File(filePath).writeAsBytesSync(img.encodeJpg(image));
  }

  // Helper methods
  void _checkEqualLength(List<double> xValues, List<double> yValues) {
    if (xValues.length != yValues.length) {
      throw ArgumentError('xValues and yValues must have the same length');
    }
  }

  List<double> _getMinMax(List<double> xValues, List<double> yValues) {
    final minX = xValues.reduce(min);
    final maxX = xValues.reduce(max);
    final minY = yValues.reduce(min);
    final maxY = yValues.reduce(max);
    return [minX, maxX, minY, maxY];
  }

  void _drawAxes(img.Image image, double minX, double maxX, double minY, double maxY) {
    _drawLine(image, padding, height - padding, width - padding, height - padding, img.ColorRgb8(0, 0, 0)); // X-axis
    _drawLine(image, padding, padding, padding, height - padding, img.ColorRgb8(0, 0, 0)); // Y-axis
  }

  void _addTitleAndLabels(img.Image image, double minX, double maxX, double minY, double maxY, String title) {
    _drawSimpleText(image, 'X-axis', width ~/ 2, height - padding ~/ 2);
    _drawSimpleText(image, 'Y-axis', padding ~/ 2, height ~/ 2);

    if (title.isNotEmpty) {
      _drawSimpleText(image, title, width ~/ 2, padding ~/ 2);
    }

    _drawSimpleText(image, minX.toStringAsFixed(2), padding, height - padding ~/ 2);
    _drawSimpleText(image, maxX.toStringAsFixed(2), width - padding, height - padding ~/ 2);
    _drawSimpleText(image, minY.toStringAsFixed(2), padding ~/ 2, height - padding);
    _drawSimpleText(image, maxY.toStringAsFixed(2), padding ~/ 2, padding);
  }

  void _drawLine(img.Image image, int x1, int y1, int x2, int y2, img.Color color) {
    img.drawLine(image, x1: x1, y1: y1, x2: x2, y2: y2, color: color);
  }

  void _drawCircle(img.Image image, int x, int y, int radius, img.Color color) {
    img.fillCircle(image, x: x, y: y, radius: radius, color: color);
  }

  void _drawSimpleText(img.Image image, String text, int x, int y) {
    img.drawString(image, font: font,  text, x: x, y: y, color: img.ColorRgb8(0, 0, 0));
  }

  int _normalize(double value, double minValue, double maxValue, int scale) {
    return ((value - minValue) / (maxValue - minValue) * scale).round();
  }

  
}
