import 'package:ml_dart/ml_dart.dart';

class ModelImporter {
  // Import model from a file
  static Map<String, dynamic> importModel(String filePath) {
    return Serialization.deserializeModel(filePath);
  }
}
