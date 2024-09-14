import 'dart:convert';
import 'dart:io';

class Serialization {
  // Serialize model with weights, layers, and hyperparameters
  static void serializeModel(Map<String, dynamic> modelData, String filePath) {
    final jsonString = jsonEncode(modelData);
    final file = File(filePath);
    file.writeAsStringSync(jsonString);
    print('Model serialized to $filePath');
  }

  // Deserialize model from file
  static Map<String, dynamic> deserializeModel(String filePath) {
    final file = File(filePath);
    final jsonString = file.readAsStringSync();
    return jsonDecode(jsonString) as Map<String, dynamic>;
  }

  // Serialize dataset (features, labels)
  static void serializeDataset(Map<String, dynamic> dataset, String filePath) {
    final jsonString = jsonEncode(dataset);
    final file = File(filePath);
    file.writeAsStringSync(jsonString);
    print('Dataset serialized to $filePath');
  }

  // Deserialize dataset from file
  static Map<String, dynamic> deserializeDataset(String filePath) {
    final file = File(filePath);
    final jsonString = file.readAsStringSync();
    return jsonDecode(jsonString) as Map<String, dynamic>;
  }
}
