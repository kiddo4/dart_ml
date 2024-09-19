import 'package:dart_ml/core/tensors.dart';
import 'package:dart_ml/data_processing/data_cleaner.dart';
import 'package:dart_ml/data_processing/data_loader.dart';
import 'package:dart_ml/data_processing/data_splitter.dart';
import 'package:dart_ml/data_processing/feature_engineer.dart';
import 'package:dart_ml/model_training/confusion_matrix.dart';
import 'package:dart_ml/model_training/k-fold_cross_validation.dart';
import 'package:dart_ml/model_training/logistic_regression.dart';
import 'dart:typed_data';

void main() async {
  final dataLoader = DataLoader();
  final dataCleaner = DataCleaner();
  final dataSplitter = DataSplitter();
  final featureEngineer = FeatureEngineer();

  // Load the Titanic dataset
  final titanicData =
      await dataLoader.loadCSV('lib/data_processing/example/titanic.csv');
  print('Loaded ${titanicData.length} rows of data');

  // Clean and preprocess the data
  final cleanedData = dataCleaner.fillMissingValues(titanicData, 'Age', 30.0);
  final encodedData = dataCleaner
      .encodeCategorical(cleanedData, 'Sex', {'male': 0, 'female': 1});

  // Clean and convert the 'Fare' column
  final fareCleanedData = encodedData.map((row) {
    var fare = row['Fare'];
    if (fare is String) {
      fare = fare.replaceAll(RegExp(r'[^\d.]'), '');
      fare = double.tryParse(fare) ?? 0.0;
    } else if (fare is! num) {
      fare = 0.0;
    }
    row['Fare'] = (fare as num).toDouble();
    return row;
  }).toList();

  final normalizedData = featureEngineer.normalize(fareCleanedData, 'Fare');
  final standardizedData = featureEngineer.standardize(normalizedData, 'Age');

  // Print the keys of the first row to see what features we have
  print('Features: ${standardizedData[0].keys.toList()}');

  // Determine input size (feature size) and output size (binary classification)
  final inputSize = standardizedData[0].length - 1; // Exclude target variable
  final outputSize = 1; // Binary classification (survived or not)

  print('Input size: $inputSize');

  // Split the data into training, validation, and test sets
  final splitData = dataSplitter.trainValidationTestSplit(standardizedData,
      validationSize: 0.2, testSize: 0.2);

  // Convert data to Lists of Tensors
  print('\nConverting training data to tensors...');
  final trainX = convertToTensorList(splitData['train']!, true, inputSize);
  final trainY = convertToTensorList(splitData['train']!, false, inputSize);

  print('\nConverting validation data to tensors...');
  final valX = convertToTensorList(splitData['validation']!, true, inputSize);
  final valY = convertToTensorList(splitData['validation']!, false, inputSize);

  print('\nConverting test data to tensors...');
  final testX = convertToTensorList(splitData['test']!, true, inputSize);
  final testY = convertToTensorList(splitData['test']!, false, inputSize);

  // Perform k-fold cross-validation
  int k = 5;
  List<Map<String, Map<String, List<Tensor>>>> folds = kFoldCrossValidation(trainX + valX, trainY + valY, k);

  List<double> accuracies = [];
  List<List<List<int>>> confusionMatrices = [];
  List<double> aucScores = [];

  for (int i = 0; i < k; i++) {
    print('Fold ${i + 1}');
    final model = LogisticRegression(inputSize, outputSize, regularizationStrength: 0.01);
    
    var trainData = folds[i]['train']?['data'];
    var trainLabels = folds[i]['train']?['labels'];
    var testData = folds[i]['test']?['data'];
    var testLabels = folds[i]['test']?['labels'];

    if (trainData == null || trainLabels == null || testData == null || testLabels == null) {
      print('Error: Missing data in fold $i');
      continue;
    }
    
    model.train(trainData, trainLabels, testData, testLabels,
        epochs: 1000, initialLearningRate: 0.01, patience: 10);

    double accuracy = model.evaluate(testData, testLabels);
    accuracies.add(accuracy);

    List<Tensor> predictions = testData.map((input) => model.forward(input)).toList();
    List<List<int>> matrix = confusionMatrix(predictions, testLabels, outputSize);
    confusionMatrices.add(matrix);

    Map<String, dynamic> rocCurveData = model.rocCurve(testData, testLabels);
    double auc = model.auc(rocCurveData);
    aucScores.add(auc);

    print('Fold ${i + 1} Accuracy: $accuracy');
    print('Fold ${i + 1} AUC: $auc');
    printConfusionMatrix(matrix);

    // Print feature importance
    List<double> importance = model.featureImportance();
    print('Feature Importance:');
    for (int j = 0; j < importance.length; j++) {
      print('Feature $j: ${importance[j]}');
    }
  }

  double averageAccuracy = accuracies.reduce((a, b) => a + b) / k;
  double averageAUC = aucScores.reduce((a, b) => a + b) / k;
  print('Average Accuracy: $averageAccuracy');
  print('Average AUC: $averageAUC');

  // Train final model on all training data
  final finalModel = LogisticRegression(inputSize, outputSize, regularizationStrength: 0.01);
  finalModel.train(trainX + valX, trainY + valY, testX, testY,
      epochs: 1000, initialLearningRate: 0.01, patience: 10);

  // Evaluate on test set
  double testAccuracy = finalModel.evaluate(testX, testY);
  print('Final Test Accuracy: $testAccuracy');

  Map<String, dynamic> testRocCurveData = finalModel.rocCurve(testX, testY);
  double testAUC = finalModel.auc(testRocCurveData);
  print('Final Test AUC: $testAUC');

  // Make prediction on sample passenger
  print('Creating sample passenger for prediction...');
  final samplePassengerData = [3, 0, 22, 1, 0, 8.25, 2, 0].map((e) => e.toDouble()).toList();
  if (samplePassengerData.length != inputSize) {
    throw Exception('Sample passenger data does not match input size ($inputSize)');
  }

  final samplePassenger = Tensor([1, inputSize], Float32List.fromList(samplePassengerData));
  final prediction = finalModel.forward(samplePassenger);
  print('Survival Prediction: ${prediction.data[0] > 0.5 ? 'Survived' : 'Not Survived'}');
  print('Survival Probability: ${prediction.data[0]}');

  // Print final feature importance
  List<double> finalImportance = finalModel.featureImportance();
  print('Final Feature Importance:');
  for (int j = 0; j < finalImportance.length; j++) {
    print('Feature $j: ${finalImportance[j]}');
  }
}