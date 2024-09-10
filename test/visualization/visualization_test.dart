import 'dart:io';
import 'package:dart_ml/visualization/visualizations.dart';
import 'package:test/test.dart';
import 'package:image/image.dart' as img;
// import 'package:dart_ml/visualization/visualization.dart';

void main() {
  late Visualization visualization;

  setUp(() {
    final fontZipPath = 'fonts/arial.ttf.zip';
    final fontBytes = File(fontZipPath).readAsBytesSync();
    final font = img.BitmapFont.fromZip(fontBytes);
    visualization = Visualization(font: font);
  });

  test('drawLinePlot creates an image file', () {
    final xValues = [0.0, 1.0, 2.0, 3.0, 4.0];
    final yValues = [0.0, 1.0, 4.0, 9.0, 16.0];
    final filePath = 'test_output_line_plot.jpg';

    visualization.drawLinePlot(xValues, yValues, filePath);

    expect(File(filePath).existsSync(), true);
    File(filePath).deleteSync(); // Cleanup after test
  });

  test('drawScatterPlot creates an image file', () {
    final xValues = [0.0, 1.0, 2.0, 3.0, 4.0];
    final yValues = [0.0, 1.5, 3.5, 7.0, 8.5];
    final filePath = 'test_output_scatter_plot.jpg';

    visualization.drawScatterPlot(xValues, yValues, filePath);

    expect(File(filePath).existsSync(), true);
    File(filePath).deleteSync(); // Cleanup after test
  });

  // Add more tests as needed
}
