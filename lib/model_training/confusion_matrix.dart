import 'package:dart_ml/core/tensors.dart';

List<List<int>> confusionMatrix(List<Tensor> predictions, List<Tensor> labels, int numClasses) {
  List<List<int>> matrix = List.generate(numClasses, (_) => List<int>.filled(numClasses, 0));

  for (int i = 0; i < predictions.length; i++) {
    int predClass = argmax(predictions[i]);
    int trueClass = argmax(labels[i]);
    matrix[trueClass][predClass]++;
  }

  return matrix;
}

void printConfusionMatrix(List<List<int>> matrix) {
  print('Confusion Matrix:');
  for (var row in matrix) {
    print(row.join('\t'));
  }
}

int argmax(Tensor tensor) {
  return tensor.data.indexOf(tensor.data.reduce((a, b) => a > b ? a : b));
}