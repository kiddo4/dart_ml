import 'dart:typed_data';

import 'package:ml_dart/ml_dart.dart';

import '../autograd.dart';


void main() {
  var a = Variable(Tensor([2, 2], Float32List.fromList([1, 2, 3, 4])));
  var b = Variable(Tensor([2, 2], Float32List.fromList([2, 3, 4, 5])));
  var c = Variable(Tensor([2, 2], Float32List.fromList([3, 4, 5, 6])));

  var d = add(a, b);
  var e = multiply(d, c);
  testCrossEntropyLos();
  e.backward();

  print('Gradient of a:');
  print(a.grad!.data);

  print('Gradient of b:');
  print(b.grad!.data);

  print('Gradient of c:');
  print(c.grad!.data);

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

void testCrossEntropyLos() {
  final outputs = Tensor([2], Float32List.fromList([0.9, 0.1])); // Example prediction
  final targets = Tensor([2], Float32List.fromList([1.0, 0.0])); // Example ground truth

  final loss = crossEntropyLoss(outputs, targets);
  print('Cross-Entropy Loss: ${loss.data[0]}'); // Should be a positive value
}