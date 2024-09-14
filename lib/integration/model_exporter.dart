import 'package:dart_ml/integration/serialization.dart';

class ModelExporter {
  // Export model to a file
  static void exportModel(Map<String, dynamic> model, String filePath) {
    Serialization.serializeModel(model, filePath);
    print('Model exported to $filePath');
  }
}
