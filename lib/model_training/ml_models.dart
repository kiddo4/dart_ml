import 'dart:io';
import 'dart:math' as math;
import 'package:image/image.dart' as img;
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
  // Check if the input is 1D and reshape it to 2D
  if (input.shape.length == 1) {
    // Reshape from [inputSize] to [1, inputSize]
    input = input.reshape([1, input.shape[0]]);
  }

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
  final int stride;
  final int padding;

  Conv2D(int inChannels, int outChannels, int kernelSize, {this.stride = 1, this.padding = 0})
      : filters = Tensor.randn([kernelSize, kernelSize, inChannels, outChannels]),
        biases = Tensor.zeros([1, 1, 1, outChannels]);

  // Perform convolution
  @override
  Tensor forward(Tensor input) {
    this.input = input;
    final inputShape = input.shape;
    final batchSize = inputShape[0];
    final inChannels = inputShape[1];
    final height = inputShape[2];
    final width = inputShape[3];
    final kernelHeight = filters.shape[0];
    final kernelWidth = filters.shape[1];
    final outChannels = filters.shape[3];

    // Calculate output dimensions
    final outHeight = (height - kernelHeight + 2 * padding) ~/ stride + 1;
    final outWidth = (width - kernelWidth + 2 * padding) ~/ stride + 1;

    // Output tensor initialized to zero
    Tensor output = Tensor.zeros([batchSize, outChannels, outHeight, outWidth]);

    // Convolution operation (basic loop)
    for (int b = 0; b < batchSize; b++) {
      for (int oc = 0; oc < outChannels; oc++) {
        for (int oh = 0; oh < outHeight; oh++) {
          for (int ow = 0; ow < outWidth; ow++) {
            double sum = 0.0; // Initialize sum for this position
            for (int ic = 0; ic < inChannels; ic++) {
              for (int kh = 0; kh < kernelHeight; kh++) {
                for (int kw = 0; kw < kernelWidth; kw++) {
                  int ih = oh * stride + kh - padding;
                  int iw = ow * stride + kw - padding;

                  if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    sum += input[[b, ic, ih, iw]] * filters[[kh, kw, ic, oc]];
                  }
                }
              }
            }
            // Add bias
            output[[b, oc, oh, ow]] = sum + biases[[0, 0, 0, oc]];
          }
        }
      }
    }
    return output;
  }

  @override
void backward(Tensor gradOutput) {
  if (input == null) throw StateError('Input is null');

  final inputShape = input!.shape;
  final gradInput = Tensor.zeros(inputShape);
  final gradFilters = Tensor.zeros(filters.shape);
  final gradBiases = Tensor.zeros([filters.shape[3]]);

  for (int b = 0; b < gradOutput.shape[0]; b++) {
    for (int oc = 0; oc < gradOutput.shape[1]; oc++) {
      for (int oh = 0; oh < gradOutput.shape[2]; oh++) {
        for (int ow = 0; ow < gradOutput.shape[3]; ow++) {
          final grad = gradOutput[[b, oc, oh, ow]];
          for (int ic = 0; ic < inputShape[1]; ic++) {
            for (int kh = 0; kh < filters.shape[0]; kh++) {
              for (int kw = 0; kw < filters.shape[1]; kw++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                if (ih >= 0 && ih < inputShape[2] && iw >= 0 && iw < inputShape[3]) {
                  gradFilters[[kh, kw, ic, oc]] += input![[b, ic, ih, iw]] * grad;
                  gradInput[[b, ic, ih, iw]] += filters[[kh, kw, ic, oc]] * grad;
                }
              }
            }
          }
        }
      }
    }
  }

  this.gradFilters = gradFilters;
  this.gradBiases = gradBiases;
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

class MaxPooling implements Model {
  final int poolSize;
  final int stride;

  MaxPooling(this.poolSize, {this.stride = 2});

  @override
  Tensor forward(Tensor input) {
    final batchSize = input.shape[0];
    final channels = input.shape[1];
    final height = input.shape[2];
    final width = input.shape[3];

    final outHeight = (height - poolSize) ~/ stride + 1;
    final outWidth = (width - poolSize) ~/ stride + 1;

    Tensor output = Tensor.zeros([batchSize, channels, outHeight, outWidth]);

    for (int b = 0; b < batchSize; b++) {
      for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < outHeight; oh++) {
          for (int ow = 0; ow < outWidth; ow++) {
            double maxVal = -double.infinity;
            for (int ph = 0; ph < poolSize; ph++) {
              for (int pw = 0; pw < poolSize; pw++) {
                int ih = oh * stride + ph;
                int iw = ow * stride + pw;
                if (ih < height && iw < width) {
                  maxVal = math.max(maxVal, input[[b, c, ih, iw]]);
                }
              }
            }
            output[[b, c, oh, ow]] = maxVal;
          }
        }
      }
    }
    return output;
  }

  @override
  void backward(Tensor gradOutput) {
    // Implement gradient for max pooling if needed
  }

  @override
  void updateParameters(double learningRate) {
    // No parameters to update in MaxPooling
  }
}


class Softmax implements Model {
  @override
  Tensor forward(Tensor input) {
    final expData = input.elementwiseOperation((x) => exp(x));
    final sumExp = expData.sum(axis: 1).reshape([input.shape[0], 1]);
    return expData / sumExp; // Normalize to get probabilities
  }

  @override
  void backward(Tensor gradOutput) {
    // Backpropagation through softmax
  }

  @override
  void updateParameters(double learningRate) {
    // No parameters to update in Softmax
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



class CNNModel extends SequentialModel {
  CNNModel()
      : super([
          Conv2D(1, 32, 3),   // Conv2D Layer: 1 input channel (grayscale), 32 filters, kernel size 3x3
          ReLU(),             // ReLU Activation
          MaxPooling(2),      // MaxPooling: 2x2 window
          Conv2D(32, 64, 3),  // Conv2D Layer: 32 input channels, 64 filters, kernel size 3x3
          ReLU(),             // ReLU Activation
          MaxPooling(2),      // MaxPooling: 2x2 window
          LinearLayer(64 * 7 * 7, 128), // Flatten and pass through a fully connected layer
          ReLU(),             // ReLU Activation
          LinearLayer(128, 10), // Final layer for 10-class classification (e.g., MNIST digits)
          Softmax()            // Softmax for class probabilities
        ]);
}



// MNIST data loader
class MNISTLoader {
  List<Tensor> loadImages(String path) {
    final file = File(path);
    final bytes = file.readAsBytesSync();
    final images = <Tensor>[];
    
    for (int i = 16; i < bytes.length; i += 28 * 28) {
      final imageBytes = bytes.sublist(i, i + 28 * 28);
      final floatList = Float32List.fromList(imageBytes.map((e) => e / 255.0).toList());
      images.add(Tensor([1, 1, 28, 28], floatList));
    }
    
    return images;
  }

  List<Tensor> loadLabels(String path) {
    final file = File(path);
    final bytes = file.readAsBytesSync();
    final labels = <Tensor>[];
    
    for (int i = 8; i < bytes.length; i++) {
      final label = bytes[i];
      final oneHot = List<double>.filled(10, 0.0);
      oneHot[label] = 1.0;
      labels.add(Tensor([1, 10], Float32List.fromList(oneHot)));
    }
    
    return labels;
  }
}

// Improved CNN model for MNIST
class MNISTModel extends SequentialModel {
  MNISTModel()
      : super([
          Conv2D(1, 32, 3, padding: 1),
          ReLU(),
          MaxPooling(2),
          Conv2D(32, 64, 3, padding: 1),
          ReLU(),
          MaxPooling(2),
          LinearLayer(64 * 7 * 7, 128),
          ReLU(),
          // Dropout(0.5),
          LinearLayer(128, 10),
          Softmax()
        ]);
}

// Improved training function with validation
void trainMNISTModel(MNISTModel model, List<Tensor> trainImages, List<Tensor> trainLabels,
                     List<Tensor> valImages, List<Tensor> valLabels, int epochs, double learningRate) {
  for (int epoch = 0; epoch < epochs; epoch++) {
    
    double trainLoss = 0.0;
    int correct = 0;
    
    for (int i = 0; i < trainImages.length; i++) {
      Tensor output = model.forward(trainImages[i]);
      Tensor loss = crossEntropyLoss(output, trainLabels[i]);
      trainLoss += loss.sum().data[0];
      
      if (argmax(output) == argmax(trainLabels[i])) correct++;
      
      model.backward(loss);
      model.updateParameters(learningRate);
    }
    
    double accuracy = correct / trainImages.length;
    print("Epoch $epoch, Train Loss: ${trainLoss / trainImages.length}, Accuracy: $accuracy");
    
    // Validation
    if (epoch % 5 == 0) {
      double valLoss = 0.0;
      int valCorrect = 0;
      
      for (int i = 0; i < valImages.length; i++) {
        Tensor output = model.forward(valImages[i]);
        Tensor loss = crossEntropyLoss(output, valLabels[i]);
        valLoss += loss.sum().data[0];
        
        if (argmax(output) == argmax(valLabels[i])) valCorrect++;
      }
      
      double valAccuracy = valCorrect / valImages.length;
      print("Validation Loss: ${valLoss / valImages.length}, Accuracy: $valAccuracy");
    }
  }
}

// Helper function to get the index of the maximum value
int argmax(Tensor tensor) {
  return tensor.data.indexOf(tensor.data.reduce(max));
}

// Function to recognize a single digit
int recognizeDigit(MNISTModel model, img.Image image) {
  // Preprocess the image
  var resized = img.copyResize(image, width: 28, height: 28);
  var grayscale = img.grayscale(resized);

  // Check if grayscale.data is null
  if (grayscale.data == null || grayscale.data!.length != 28 * 28) {
    throw Exception('Grayscale data is null or has incorrect length.');
  }

  // Convert to tensor
  // Convert to tensor
// Convert to tensor
var pixels = Float32List(28 * 28);
for (int i = 0; i < 28 * 28; i++) {
  pixels[i] = grayscale.data?.buffer.asFloat32List()[i] ?? 0 / 255.0;
}
  var tensor = Tensor([1, 1, 28, 28], pixels);

  // Forward pass
  var output = model.forward(tensor);

  // Return predicted digit
  return argmax(output);
}

