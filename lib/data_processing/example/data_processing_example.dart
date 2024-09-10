// import 'dart:io';

// import 'package:dart_ml/data_processing/data_loader.dart';
// import 'package:dart_ml/data_processing/data_cleaner.dart';
// import 'package:dart_ml/data_processing/feature_engineer.dart';
// import 'package:dart_ml/data_processing/data_splitter.dart';
// import 'package:dart_ml/visualization/visualizations.dart';
// import 'package:image/image.dart' as img;

// void main() async {
//   final dataLoader = DataLoader();
//   final dataCleaner = DataCleaner();
//   final featureEngineer = FeatureEngineer();
//   final dataSplitter = DataSplitter();

//   // Path to your font zip file
//   final fontZipPath = 'fonts/arial.ttf.zip';

//   // Read the zip file as bytes
//   final bytes = File(fontZipPath).readAsBytesSync();

//   // Load the font from the zip bytes
//   final font = img.BitmapFont.fromZip(bytes);

//   // Now you can use the font in your visualization
//   final visualization = Visualization(font: font);

//   // final xValues = [0.0, 1.0, 2.0, 3.0, 4.0];
//   // final yValues = [0.0, 1.0, 4.0, 9.0, 16.0];

 
//   // Step 1: Load the Titanic dataset
//   final titanicData = await dataLoader.loadCSV('lib/data_processing/example/titanic.csv');

//   // Step 2: Clean the data
//   final cleanedData = dataCleaner.fillMissingValues(titanicData, 'Age', 30.0);
//   final encodedData = dataCleaner.encodeCategorical(cleanedData, 'Sex', {'male': 0, 'female': 1});

//   // Step 3: Feature Engineering
//   final normalizedData = featureEngineer.normalize(encodedData, 'Fare');
//   final standardizedData = featureEngineer.standardize(normalizedData, 'Age');

//   // Step 4: Split the data into training and test sets
//   final splitData = dataSplitter.trainTestSplit(standardizedData, 0.2);
//   final xValues = splitData['train']);
//    visualization.drawLinePlot(xValues, yValues, 'output.jpg', title: 'Quadratic Plot');
//   print('Training data sample: ${splitData['train']!.take(5)}');
//   print('Test data sample: ${splitData['test']!.take(5)}');
// }
