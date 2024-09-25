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
