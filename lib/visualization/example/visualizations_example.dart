import 'dart:io';
import 'package:dart_ml/visualization/visualizations.dart';
import 'package:image/image.dart' as img;


void main() {
  // Load the font from a zip file
  final fontZipPath = 'fonts/arial.ttf.zip';  // Replace with the actual path
  final fontBytes = File(fontZipPath).readAsBytesSync();
  final font = img.BitmapFont.fromZip(fontBytes);

  // Create a Visualization instance
  final visualization = Visualization(font: font);

  // Line Plot Example
  final linePlotXValues = [0.0, 1.0, 2.0, 3.0, 4.0];
  final linePlotYValues = [0.0, 1.0, 4.0, 9.0, 16.0];
  final linePlotFilePath = 'output_line_plot.jpg';

  visualization.drawLinePlot(
    linePlotXValues,
    linePlotYValues,
    linePlotFilePath,
    title: 'Quadratic Function'
  );

  print('Line plot saved to $linePlotFilePath');

  // Scatter Plot Example
  final scatterPlotXValues = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
  final scatterPlotYValues = [0.0, 1.5, 3.5, 7.0, 8.5, 15.0];
  final scatterPlotFilePath = 'output_scatter_plot.jpg';

  visualization.drawScatterPlot(
    scatterPlotXValues,
    scatterPlotYValues,
    scatterPlotFilePath,
    title: 'Scatter Plot Example'
  );

  print('Scatter plot saved to $scatterPlotFilePath');
}
