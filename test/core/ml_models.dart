import 'package:test/test.dart';
import 'package:ml_dart/ml_dart.dart';

void main() {
  test('Linear Layer Forward Pass', () {
    final layer = LinearLayer(10, 20);
    final input = Tensor.randn([32, 10]);
    final output = layer.forward(input);

    expect(output.shape, equals([32, 20]));
  });

  test('Sequential Model Forward and Backward Pass', () {
    final model = SequentialModel([
      LinearLayer(10, 20),
      LinearLayer(20, 1),
    ]);

    final input = Tensor.randn([32, 10]);
    final target = Tensor.randn([32, 1]);

    // Forward pass
    final output = model.forward(input);
    expect(output.shape, equals([32, 1]));

    // Compute loss (MSE)
    final loss = output.mse(target);

    // Backward pass
    final gradOutput = output - target;
    model.backward(gradOutput);

    // Update parameters
    model.updateParameters(0.01);

    expect(loss, isNotNull);
  });
}
