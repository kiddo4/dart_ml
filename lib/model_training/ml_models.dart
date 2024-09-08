
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
    // print('Input shape: ${input.shape}');
    Tensor output = input.matmul(weights);
    // print('Output shape before bias: ${output.shape}');
    Tensor broadcastedBias = bias.broadcastTo(output.shape);
    // print('Broadcasted Bias shape: ${broadcastedBias.shape}');
    return output + broadcastedBias;
  }

  @override
  void backward(Tensor gradOutput) {
    if (input == null) throw StateError('Input is null');
    gradWeights = input!.transpose().matmul(gradOutput);
    gradBias = gradOutput.sum(axis: 0);
  }

  @override
  void updateParameters(double learningRate) {
    if (gradWeights == null || gradBias == null)
      throw StateError('Gradients are null');

    // Update weights
    final weightUpdate =
        gradWeights!.elementwiseOperation((x) => x * learningRate);
    weights = weights - weightUpdate;

    // Update bias
    final biasUpdate = gradBias!.elementwiseOperation((x) => x * learningRate);
    bias = bias - biasUpdate;
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
