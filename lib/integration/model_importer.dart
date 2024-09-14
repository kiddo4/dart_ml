import 'package:dart_ml/integration/serialization.dart';

class ModelImporter {
  // Import model from a file
  static Map<String, dynamic> importModel(String filePath) {
    return Serialization.deserializeModel(filePath);
  }
}
