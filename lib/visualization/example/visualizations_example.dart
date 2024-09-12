import 'package:dart_ml/visualization/visualizations.dart';
import 'package:image/image.dart' as img;
import 'dart:io';

void main() {
  final fontBytes = File('fonts/arial.ttf.zip').readAsBytesSync();
  final font = img.BitmapFont.fromZip(fontBytes); // Make sure to load an actual BitmapFont

  final visualization = Visualization(
    width: 800,
    height: 600,
    padding: 40,
    font: font,
  );

  // Example data
  final xValues = [1.0, 2.0, 3.0, 4.0, 5.0];
  final yValues = [2.0, 3.5, 1.5, 4.5, 3.0];
  
  final heatmapData = List.generate(
    10,
    (y) => List.generate(10, (x) => (x + y) / 20),
  );

  // Generate line plot
  visualization.drawLinePlot(
    xValues,
    yValues,
    'line_plot.jpg',
    title: 'Line Plot Example',
  );

  // Generate scatter plot
  visualization.drawScatterPlot(
    xValues,
    yValues,
    'scatter_plot.jpg',
    title: 'Scatter Plot Example',
  );

 
  // Generate heatmap
  visualization.drawHeatmap(
    heatmapData,
    'heatmap.jpg',
  );

  print('Visualizations generated successfully!');
}
