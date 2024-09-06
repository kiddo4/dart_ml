import 'package:dart_ml/core/matrices.dart';

class SGD {
  final double learningRate;

  SGD({this.learningRate = 0.01});

  void updateWeights(Matrix weights, Matrix gradients) {
    for (int i = 0; i < weights.data.length; i++) {
      weights.data[i] -= learningRate * gradients.data[i];
    }
  }
}
