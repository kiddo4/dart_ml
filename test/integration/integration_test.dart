import 'dart:io';


import 'package:test/test.dart';
import 'package:dart_ml/integration/serialization.dart';
import 'package:dart_ml/integration/model_exporter.dart';
import 'package:dart_ml/integration/model_importer.dart';

void main() {
  group('Model Exporter, Model Importer, and Serialization Tests', () {
    final String modelFilePath = 'test_model.json';
    final String datasetFilePath = 'test_dataset.json';
    
    final Map<String, dynamic> testModel = {
      'type': 'SequentialModel',
      'layers': [
        {'type': 'Dense', 'units': 64, 'activation': 'relu'},
        {'type': 'Dense', 'units': 10, 'activation': 'softmax'},
      ],
    };

    final Map<String, dynamic> testDataset = {
      'features': [
        [1.0, 2.0],
        [3.0, 4.0],
      ],
      'labels': [0, 1],
    };

    test('Model Export and Import', () async {
      // Export model
      ModelExporter.exportModel(testModel, modelFilePath);

      // Import model
      final importedModel = ModelImporter.importModel(modelFilePath);

      // Verify if the imported model matches the original
      expect(importedModel, equals(testModel));

      // Clean up test files
      File(modelFilePath).deleteSync();
    });

    test('Serialization and Deserialization of Model', () {
      // Serialize model
      Serialization.serializeModel(testModel, modelFilePath);

      // Deserialize model
      final deserializedModel = Serialization.deserializeModel(modelFilePath);

      // Verify if the deserialized model matches the original
      expect(deserializedModel, equals(testModel));

      // Clean up test files
      File(modelFilePath).deleteSync();
    });

    test('Serialization and Deserialization of Dataset', () {
      // Serialize dataset
      Serialization.serializeDataset(testDataset, datasetFilePath);

      // Deserialize dataset
      final deserializedDataset = Serialization.deserializeDataset(datasetFilePath);

      // Verify if the deserialized dataset matches the original
      expect(deserializedDataset, equals(testDataset));

      // Clean up test files
      File(datasetFilePath).deleteSync();
    });
  });
}
