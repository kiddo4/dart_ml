import 'package:test/test.dart';
import 'package:dart_ml/data_processing/data_loader.dart';
import 'package:dart_ml/data_processing/data_cleaner.dart';
import 'package:dart_ml/data_processing/feature_engineer.dart';
import 'package:dart_ml/data_processing/data_splitter.dart';

void main() {
  group('Data Processing Tests', () {
    test('CSV loading', () async {
      final dataLoader = DataLoader();
      final data = await dataLoader.loadCSV('lib/data_processing/example/titanic.csv');
      expect(data.isNotEmpty, true);
    });

    test('Data cleaning - filling missing values', () {
      final dataCleaner = DataCleaner();
      final data = [
        {'Age': 25.0},
        {'Age': null}, // Test with a null value
      ];
      final cleanedData = dataCleaner.fillMissingValues(data, 'Age', 30.0);
      expect(cleanedData[1]['Age'], 30.0); // Expect the missing value to be filled with 30.0
    });

    test('Data cleaning - encoding categorical', () {
      final dataCleaner = DataCleaner();
      final data = [
        {'Sex': 'male'},
        {'Sex': 'female'},
      ];
      final encodedData = dataCleaner.encodeCategorical(data, 'Sex', {'male': 0, 'female': 1});
      expect(encodedData[0]['Sex'], 0); // Expect integer 0
      expect(encodedData[1]['Sex'], 1); // Expect integer 1
    });

    test('Feature engineering - normalization', () {
      final featureEngineer = FeatureEngineer();
      final data = [
        {'Fare': 10.0},
        {'Fare': 20.0},
        {'Fare': 30.0},
      ];
      final normalizedData = featureEngineer.normalize(data, 'Fare');
      expect(normalizedData[0]['Fare'], 0.0);
      expect(normalizedData[2]['Fare'], 1.0);
    });

    test('Feature engineering - standardization', () {
      final featureEngineer = FeatureEngineer();
      final data = [
        {'Age': 20.0},
        {'Age': 30.0},
        {'Age': 40.0},
      ];
      final standardizedData = featureEngineer.standardize(data, 'Age');
      expect(standardizedData[1]['Age'], closeTo(0.0, 0.1)); // mean should be around 0
    });

    test('Data splitting', () {
      final dataSplitter = DataSplitter();
      final data = List.generate(100, (index) => {'feature': index});
      final splitData = dataSplitter.trainTestSplit(data, 0.2);
      expect(splitData['train']!.length, 80); // Test for 80% train data
      expect(splitData['test']!.length, 20);  // Test for 20% test data
    });
  });
}

