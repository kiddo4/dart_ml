import 'dart:math';

import 'package:ml_dart/ml_dart.dart';
import 'package:image/image.dart' as img;
import 'dart:io';

void main() {
  final fontBytes = File('fonts/arial.ttf.zip').readAsBytesSync();
  final font = img.BitmapFont.fromZip(fontBytes); // Make sure to load an actual BitmapFont

  // Create an instance of Visualization
  final visualizer = Visualization(
    width: 700,
    height: 500,
    padding: 60,
    font: font,
    zoomScale: 1.0,
    panX: 0.0,
    panY: 0.0,
    title: 'My Visualization',
    xAxisLabel: 'Epoch',
    yAxisLabel: 'Loss',
  );

  // Line Plot Example
  final xValuesLine = List.generate(100, (index) => index.toDouble());
  final yValuesLine = List.generate(100, (index) => sin(index * 0.1) * 10 + 50);
  visualizer.drawLinePlot(xValuesLine, yValuesLine, 'line_plot.jpg', title: 'Sine Wave Line Plot', downsampleRate: 2);

  // Scatter Plot Example
  final xValuesScatter = List.generate(100, (index) => index.toDouble());
  final yValuesScatter = List.generate(100, (index) => cos(index * 0.1) * 10 + 50);
  visualizer.drawScatterPlot(xValuesScatter, yValuesScatter, 'scatter_plot.jpg', title: 'Cosine Scatter Plot', downsampleRate: 2);

  // Heatmap Example
  final heatmapData = List.generate(50, (y) => List.generate(50, (x) => (x * y) / 2500));
  visualizer.drawHeatmap(heatmapData, 'heatmap.jpg', title: 'Sample Heatmap');

  // Loss Curve Example
  final losses = List.generate(100, (index) => exp(-index / 10) + (Random().nextDouble() * 0.1));
  visualizer.drawLossCurve(losses, 'loss_curve.jpg', title: 'Training Loss Curve');

  print('Visualizations generated successfully!');
}

  

