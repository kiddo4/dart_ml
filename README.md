

# Dart ML - Machine Learning Library for Dart

Dart ML is a machine learning library built to provide efficient tools for data processing, model training, and visualization, all within the Dart ecosystem. This project aims to make it easier for developers and data scientists to perform machine learning tasks directly in Dart.

## Features

- **Data Processing:** Tools for data loading, cleaning, and feature engineering.
- **Model Training:** Support for various machine learning models, including logistic regression and neural networks.
- **Visualization:** Plot graphs, visualize training progress, and generate heatmaps.
- **Cross-validation:** Perform k-fold cross-validation and model evaluation metrics like confusion matrices and AUC scores.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Data Processing Example](#data-processing-example)
  - [Model Training Example](#model-training-example)
  - [Visualization Example](#visualization-example)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use the Dart ML library, add the following to your `pubspec.yaml` file:

```yaml
dependencies:
  ml_dart:
    git: https://github.com/your-repo/ml_dart.git
```

Then run:

```bash
dart pub get
```

## Usage

### Data Processing Example

This example demonstrates how to load, clean, engineer features, and split the Titanic dataset.

```dart
import 'package:ml_dart/ml_dart.dart';

void main() async {
  final dataLoader = DataLoader();
  final dataCleaner = DataCleaner();
  final featureEngineer = FeatureEngineer();
  final dataSplitter = DataSplitter();

  // Step 1: Load the Titanic dataset
  final titanicData = await dataLoader.loadCSV('lib/data_processing/example/titanic.csv');

  // Step 2: Clean the data
  final cleanedData = dataCleaner.fillMissingValues(titanicData, 'Age', 30.0);
  final encodedData = dataCleaner.encodeCategorical(cleanedData, 'Sex', {'male': 0, 'female': 1});

  // Step 3: Feature Engineering
  final normalizedData = featureEngineer.normalize(encodedData, 'Fare');
  final standardizedData = featureEngineer.standardize(normalizedData, 'Age');

  // Step 4: Split the data into training and test sets
  final splitData = dataSplitter.trainTestSplit(standardizedData, testSize: 0.2);

  print('Training data sample: ${splitData['train']!.take(5)}');
  print('Test data sample: ${splitData['test']!.take(5)}');
}
```

### Model Training Example

This example demonstrates training a neural network on the MNIST dataset and saving a trained model.

```dart
import 'dart:io';
import 'package:ml_dart/ml_dart.dart';
import 'package:image/image.dart' as img;

void main() async {
  final loader = MNISTLoader();
  final trainImages = loader.loadImages('assets/train-images.idx3-ubyte');
  final trainLabels = loader.loadLabels('assets/train-labels.idx1-ubyte');
  final testImages = loader.loadImages('assets/t10k-images.idx3-ubyte');
  final testLabels = loader.loadLabels('assets/t10k-labels.idx1-ubyte');

  final model = MNISTModel();
  
  // Train the model
  trainMNISTModel(model, trainImages, trainLabels, testImages, testLabels, 50, 0.001);

  // Example usage: recognize a digit from a file
  final image = img.decodeImage(File('assets/sample_digit.png').readAsBytesSync())!;
  final prediction = recognizeDigit(model, image);
  print('Predicted digit: $prediction');
}
```

### Visualization Example

This example demonstrates how to generate line plots, scatter plots, heatmaps, and loss curves.

```dart
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
```

## Contributing

Feel free to contribute to this project! Please submit issues and pull requests through our [GitHub repository](https://github.com/your-repo/ml_dart).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.