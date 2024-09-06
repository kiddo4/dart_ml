import 'dart:math' as math;
import 'package:dart_ml/core/tensors.dart';

// Abstract base class for machine learning models
abstract class Model {
  Tensor forward(Tensor input);
  void backward(Tensor gradOutput);
  void updateParameters(double learningRate);
}

// Linear layer implementation
class LinearLayer implements Model {
  Tensor weights;
  Tensor bias;
  Tensor? input;
  Tensor? gradWeights;
  Tensor? gradBias;

  LinearLayer(int inputSize, int outputSize)
      : weights = Tensor.randn([inputSize, outputSize]),
        bias = Tensor.zeros([1, outputSize]);

  @override
  Tensor forward(Tensor input) {
    this.input = input;
    return input.matmul(weights) + bias;
  }

  @override
  void backward(Tensor gradOutput) {
    if (input == null) throw StateError('Input is null');
    gradWeights = input!.transpose().matmul(gradOutput);
    gradBias = gradOutput.sum(axis: 0);
  }

  @override
  void updateParameters(double learningRate) {
    if (gradWeights == null || gradBias == null) throw StateError('Gradients are null');
    // Ensure the operations are performed element-wise
    weights = weights - (gradWeights! * learningRate);
    bias = bias - (gradBias! * learningRate);
  }
}

// Sequential model implementation for stacking layers
class SequentialModel implements Model {
  final List<Model> layers;

  SequentialModel(this.layers);

  @override
  Tensor forward(Tensor input) {
    Tensor output = input;
    for (var layer in layers) {
      output = layer.forward(output);
    }
    return output;
  }

  @override
  void backward(Tensor gradOutput) {
    Tensor currentGrad = gradOutput;
    for (var layer in layers.reversed) {
      layer.backward(currentGrad);
      // Compute gradients for the input for each layer
      // Note: Actual gradient propagation for input is not implemented here
    }
  }

  @override
  void updateParameters(double learningRate) {
    for (var layer in layers) {
      layer.updateParameters(learningRate);
    }
  }
}

// Main function for training the model
void main() {
  final model = SequentialModel([
    LinearLayer(10, 20),
    LinearLayer(20, 1),
  ]);

  final input = Tensor.randn([32, 10]); // Batch of 32 samples, 10 features each
  final target = Tensor.randn([32, 1]);

  final learningRate = 0.01;
  final numEpochs = 100;

  for (var epoch = 0; epoch < numEpochs; epoch++) {
    // Forward pass
    final output = model.forward(input);

    // Compute loss (Mean Squared Error in this example)
    final loss = output.mse(target);

    // Backward pass
    final gradOutput = output - target; // Gradient of MSE
    model.backward(gradOutput);

    // Update parameters
    model.updateParameters(learningRate);

    if (epoch % 10 == 0) {
      print('Epoch $epoch, Loss: $loss');
    }
  }
}
