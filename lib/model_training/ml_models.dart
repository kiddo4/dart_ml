import 'dart:math' as math;
import 'dart:math';
import 'dart:typed_data';
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
    Tensor output = input.matmul(weights);
    Tensor broadcastedBias = bias.broadcastTo(output.shape);
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
    if (gradWeights == null || gradBias == null) throw StateError('Gradients are null');

    final weightUpdate = gradWeights!.elementwiseOperation((x) => x * learningRate);
    weights = weights - weightUpdate;

    final biasUpdate = gradBias!.elementwiseOperation((x) => x * learningRate);
    bias = bias - biasUpdate;
  }
}

// ReLU activation
class ReLU implements Model {
  @override
  Tensor forward(Tensor input) {
    return input.elementwiseOperation((x) => x > 0 ? x : 0);
  }

  @override
  void backward(Tensor gradOutput) {
    // No learnable parameters to update
  }

  @override
  void updateParameters(double learningRate) {
    // No learnable parameters to update
  }
}

// Sigmoid activation
class Sigmoid implements Model {
  @override
  Tensor forward(Tensor input) {
    return input.elementwiseOperation((x) => 1 / (1 + exp(-x)));
  }

  @override
  void backward(Tensor gradOutput) {
    // No learnable parameters to update
  }

  @override
  void updateParameters(double learningRate) {
    // No learnable parameters to update
  }
}

// Tanh activation
class Tanh implements Model {
  @override
  Tensor forward(Tensor input) {
    return input.elementwiseOperation((x) => tan(x));
  }

  @override
  void backward(Tensor gradOutput) {
    // No learnable parameters to update
  }

  @override
  void updateParameters(double learningRate) {
    // No learnable parameters to update
  }
}

// Conv2D layer implementation (placeholder)
class Conv2D implements Model {
  Tensor filters;
  Tensor biases;
  Tensor? input;
  Tensor? gradFilters;
  Tensor? gradBiases;

  Conv2D(int inChannels, int outChannels, int kernelSize)
      : filters = Tensor.randn([kernelSize, kernelSize, inChannels, outChannels]),
        biases = Tensor.zeros([1, 1, 1, outChannels]);

  @override
  Tensor forward(Tensor input) {
    this.input = input;
    // Implement convolution operation here (e.g., using im2col and matrix multiplication)
    return input; // Placeholder
  }

  @override
  void backward(Tensor gradOutput) {
    if (input == null) throw StateError('Input is null');
    gradFilters = Tensor.zeros(filters.shape);
    gradBiases = gradOutput.sum(axis: 0);
  }

  @override
  void updateParameters(double learningRate) {
    if (gradFilters == null || gradBiases == null) throw StateError('Gradients are null');

    filters = filters - gradFilters!.elementwiseOperation((x) => x * learningRate);
    biases = biases - gradBiases!.elementwiseOperation((x) => x * learningRate);
  }
}




class Dropout {
  final double rate;
  Tensor? mask;

  Dropout(this.rate);

  Tensor forward(Tensor input) {
    final dropoutMask = Tensor.ones(input.shape);
    final rng = math.Random();
    
    // Create dropout mask with correct shape and values
    for (var i = 0; i < dropoutMask.data.length; i++) {
      dropoutMask.data[i] = rng.nextDouble() > rate ? 1.0 : 0.0;
    }

    mask = dropoutMask;
    
    // Apply dropout mask to the input tensor
    final newData = Float32List(input.length);
    for (var i = 0; i < input.length; i++) {
      newData[i] = input.data[i] * dropoutMask.data[i];
    }
    
    return Tensor(input.shape, newData);
  }

  void backward(Tensor gradOutput) {
    // No additional operations needed for dropout
  }

  void updateParameters(double learningRate) {
    // No parameters to update in dropout
  }
}



// Batch Normalization layer (placeholder)
class BatchNorm implements Model {
  Tensor gamma;
  Tensor beta;
  Tensor? runningMean;
  Tensor? runningVar;
  Tensor? input;
  Tensor? gradGamma;
  Tensor? gradBeta;

  BatchNorm(int numFeatures)
      : gamma = Tensor.ones([1, 1, 1, numFeatures]),
        beta = Tensor.zeros([1, 1, 1, numFeatures]),
        runningMean = Tensor.zeros([1, 1, 1, numFeatures]),
        runningVar = Tensor.ones([1, 1, 1, numFeatures]);

  @override
  Tensor forward(Tensor input) {
    this.input = input;
    // Implement Batch Normalization operation here (e.g., mean and variance calculation)
    return input; // Placeholder
  }

  @override
  void backward(Tensor gradOutput) {
    if (input == null) throw StateError('Input is null');
    gradGamma = Tensor.zeros(gamma.shape);
    gradBeta = Tensor.zeros(beta.shape);
  }

  @override
  void updateParameters(double learningRate) {
    if (gradGamma == null || gradBeta == null) throw StateError('Gradients are null');

    gamma = gamma - gradGamma!.elementwiseOperation((x) => x * learningRate);
    beta = beta - gradBeta!.elementwiseOperation((x) => x * learningRate);
  }
}

// Sequential model for stacking layers
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
