import 'dart:typed_data';
import 'package:dart_ml/core/tensors.dart';
import 'package:dart_ml/data_processing/data_cleaner.dart';
import 'package:dart_ml/data_processing/data_loader.dart';
import 'package:dart_ml/data_processing/data_splitter.dart';
import 'package:dart_ml/data_processing/feature_engineer.dart';
import 'package:dart_ml/model_training/confusion_matrix.dart';
import 'package:dart_ml/model_training/k-fold_cross_validation.dart';
import 'package:dart_ml/model_training/logistic_regression.dart';

void main() async {
  // Initialize data processing and model training components
  final dataLoader = DataLoader();
  final dataCleaner = DataCleaner();
  final dataSplitter = DataSplitter();
  final featureEngineer = FeatureEngineer();

  // Load and preprocess the Titanic dataset
  final titanicData = await dataLoader.loadCSV('lib/data_processing/example/titanic.csv');
  print('Loaded ${titanicData.length} rows of data');

  var cleanedData = dataCleaner.fillMissingValues(titanicData, 'Age', 30.0);
  cleanedData = dataCleaner.encodeCategorical(cleanedData, 'Sex', {'male': 0, 'female': 1});

  cleanedData = cleanedData.map((row) {
    var fare = row['Fare'];
    if (fare is String) {
      fare = double.tryParse(fare.replaceAll(RegExp(r'[^\d.]'), '')) ?? 0.0;
    } else if (fare is! num) {
      fare = 0.0;
    }
    row['Fare'] = (fare as num).toDouble();
    return row;
  }).toList();

  var normalizedData = featureEngineer.normalize(cleanedData, 'Fare');
  var standardizedData = featureEngineer.standardize(normalizedData, 'Age');

  print('Features: ${standardizedData[0].keys.toList()}');

  // Define input and output sizes
  final inputSize = standardizedData[0].length - 1;
  final outputSize = 1; // Binary classification

  print('Input size: $inputSize');

  // Split data into training, validation, and test sets
  final splitData = dataSplitter.trainValidationTestSplit(standardizedData, validationSize: 0.2, testSize: 0.2);

  // Convert data to Tensors
  final trainX = convertToTensorList(splitData['train']!, true, inputSize);
  final trainY = convertToTensorList(splitData['train']!, false, inputSize);
  final valX = convertToTensorList(splitData['validation']!, true, inputSize);
  final valY = convertToTensorList(splitData['validation']!, false, inputSize);
  final testX = convertToTensorList(splitData['test']!, true, inputSize);
  final testY = convertToTensorList(splitData['test']!, false, inputSize);

  // Perform k-fold cross-validation
  const k = 5;
  final folds = kFoldCrossValidation(trainX + valX, trainY + valY, k);

  final accuracies = <double>[];
  final confusionMatrices = <List<List<int>>>[];
  final aucScores = <double>[];

  for (var i = 0; i < k; i++) {
    print('Fold ${i + 1}');

    final model = LogisticRegression(inputSize, outputSize, );

    var fold = folds[i];
    final trainData = fold['train']?['data'];
    final trainLabels = fold['train']?['labels'];
    final testData = fold['test']?['data'];
    final testLabels = fold['test']?['labels'];

    if (trainData == null || trainLabels == null || testData == null || testLabels == null) {
      print('Error: Missing data in fold $i');
      continue;
    }

    // Train and evaluate the model
    model.train(trainData, trainLabels, testData, testLabels, epochs: 1000, learningRate: 0.01, patience: 10);

    final accuracy = model.evaluate(testData, testLabels);
    accuracies.add(accuracy);

    final predictions = testData.map((input) => model.forward(input)).toList();
    final matrix = confusionMatrix(predictions, testLabels, outputSize);
    confusionMatrices.add(matrix);

    final rocCurveData = model.rocCurve(testData, testLabels);
    final auc = model.auc(rocCurveData);
    aucScores.add(auc);

    print('Fold ${i + 1} Accuracy: $accuracy');
    print('Fold ${i + 1} AUC: $auc');
    printConfusionMatrix(matrix);

    final importance = model.featureImportance();
    print('Feature Importance:');
    importance.asMap().forEach((index, value) => print('Feature $index: $value'));
  }

  // Report average results
  final averageAccuracy = accuracies.reduce((a, b) => a + b) / k;
  final averageAUC = aucScores.reduce((a, b) => a + b) / k;
  print('Average Accuracy: $averageAccuracy');
  print('Average AUC: $averageAUC');

  // Train final model on all training data and evaluate
  final finalModel = LogisticRegression(inputSize, outputSize, );
  finalModel.train(trainX + valX, trainY + valY, testX, testY, epochs: 1000, learningRate: 0.01, patience: 10);

  final testAccuracy = finalModel.evaluate(testX, testY);
  print('Final Test Accuracy: $testAccuracy');

  final testRocCurveData = finalModel.rocCurve(testX, testY);
  final testAUC = finalModel.auc(testRocCurveData);
  print('Final Test AUC: $testAUC');

  // Make prediction for a sample passenger
  print('Creating sample passenger for prediction...');
  final samplePassengerData = [3, 0, 22, 1, 0, 8.25, 2, 0].map((e) => e.toDouble()).toList();
  if (samplePassengerData.length != inputSize) {
    throw Exception('Sample passenger data does not match input size ($inputSize)');
  }

  final samplePassenger = Tensor([1, inputSize], Float32List.fromList(samplePassengerData));
  final prediction = finalModel.forward(samplePassenger);
  print('Prediction (0 = did not survive, 1 = survived): ${prediction.data[0]}');
}


