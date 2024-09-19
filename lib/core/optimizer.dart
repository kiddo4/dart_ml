import 'package:dart_ml/core/matrices.dart';
import 'dart:math' as math;

abstract class Optimizer {
  void initialize(int layers, int inputSize, int outputSize);
  void updateWeights(Matrix weights, Matrix gradients, int layerIndex);
}

class Adam implements Optimizer {
 final double learningRate;
  final double beta1;
  final double beta2;
  final double epsilon;

  List<Matrix> m; // First moment vector
  List<Matrix> v; // Second moment vector
  int t; // Time step

  Adam({
    this.learningRate = 0.001,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.epsilon = 1e-8,
  }) : m = [], v = [], t = 0;

  void initialize(int layers, int inputSize, int outputSize) {
    m = List.generate(layers, (_) => Matrix.zeros(inputSize, outputSize));
    v = List.generate(layers, (_) => Matrix.zeros(inputSize, outputSize));
  }

  void updateWeights(Matrix weights, Matrix gradients, int layerIndex) {
    t += 1;

    // Update biased first moment estimate
    for (int i = 0; i < weights.data.length; i++) {
      m[layerIndex].data[i] = beta1 * m[layerIndex].data[i] + (1 - beta1) * gradients.data[i];
    }

    // Update biased second raw moment estimate
    for (int i = 0; i < weights.data.length; i++) {
      v[layerIndex].data[i] = beta2 * v[layerIndex].data[i] + (1 - beta2) * math.pow(gradients.data[i], 2);
    }

    // Compute bias-corrected first moment estimate
    final mHat = m[layerIndex].map((x) => x / (1 - math.pow(beta1, t)));

    // Compute bias-corrected second raw moment estimate
    final vHat = v[layerIndex].map((x) => x / (1 - math.pow(beta2, t)));

    // Update weights
    for (int i = 0; i < weights.data.length; i++) {
      weights.data[i] -= learningRate * mHat.data[i] / (math.sqrt(vHat.data[i]) + epsilon);
    }
  }
}

class SGD implements Optimizer {
  final double learningRate;

  SGD({this.learningRate = 0.01});

  @override
  void initialize(int layers, int inputSize, int outputSize) {
    // No initialization needed for basic SGD
  }

  @override
  void updateWeights(Matrix weights, Matrix gradients, int layerIndex) {
    for (int i = 0; i < weights.data.length; i++) {
      weights.data[i] -= learningRate * gradients.data[i];
    }
  }
}