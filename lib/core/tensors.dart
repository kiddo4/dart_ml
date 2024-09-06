import 'dart:math' as math;
import 'dart:typed_data';

class Tensor {
  final List<int> shape;
  final Float32List data;

  Tensor(this.shape, this.data) {
    if (shape.reduce((a, b) => a * b) != data.length) {
      throw ArgumentError('Shape does not match data length');
    }
  }

  factory Tensor.zeros(List<int> shape) {
    final size = shape.reduce((a, b) => a * b);
    return Tensor(shape, Float32List(size));
  }

  factory Tensor.ones(List<int> shape) {
    final size = shape.reduce((a, b) => a * b);
    return Tensor(shape, Float32List(size)..fillRange(0, size, 1.0));
  }

  factory Tensor.randn(List<int> shape) {
    final size = shape.reduce((a, b) => a * b);
    final data = Float32List(size);
    final random = math.Random();
    for (var i = 0; i < size; i++) {
      data[i] = _boxMullerTransform(random);
    }
    return Tensor(shape, data);
  }

  static double _boxMullerTransform(math.Random random) {
    final u1 = 1.0 - random.nextDouble();
    final u2 = 1.0 - random.nextDouble();
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2);
  }

  Tensor reshape(List<int> newShape) {
    if (newShape.reduce((a, b) => a * b) != data.length) {
      throw ArgumentError('New shape does not match data length');
    }
    return Tensor(newShape, data);
  }

  Tensor operator +(Tensor other) {
    if (!_shapeMatch(other)) {
      throw ArgumentError('Tensor shapes do not match for addition');
    }
    final newData = Float32List(data.length);
    for (var i = 0; i < data.length; i++) {
      newData[i] = data[i] + other.data[i];
    }
    return Tensor(shape, newData);
  }

  Tensor operator -(Tensor other) {
    if (!_shapeMatch(other)) {
      throw ArgumentError('Tensor shapes do not match for subtraction');
    }
    final newData = Float32List(data.length);
    for (var i = 0; i < data.length; i++) {
      newData[i] = data[i] - other.data[i];
    }
    return Tensor(shape, newData);
  }

  Tensor operator *(Tensor other) {
    if (!_shapeMatch(other)) {
      throw ArgumentError('Tensor shapes do not match for element-wise multiplication');
    }
    final newData = Float32List(data.length);
    for (var i = 0; i < data.length; i++) {
      newData[i] = data[i] * other.data[i];
    }
    return Tensor(shape, newData);
  }

  Tensor matmul(Tensor other) {
    if (shape.length != 2 || other.shape.length != 2 || shape[1] != other.shape[0]) {
      throw ArgumentError('Invalid shapes for matrix multiplication');
    }
    final m = shape[0];
    final n = other.shape[1];
    final k = shape[1];
    final newData = Float32List(m * n);
    for (var i = 0; i < m; i++) {
      for (var j = 0; j < n; j++) {
        var sum = 0.0;
        for (var l = 0; l < k; l++) {
          sum += data[i * k + l] * other.data[l * n + j];
        }
        newData[i * n + j] = sum;
      }
    }
    return Tensor([m, n], newData);
  }

  bool _shapeMatch(Tensor other) {
    if (shape.length != other.shape.length) return false;
    for (var i = 0; i < shape.length; i++) {
      if (shape[i] != other.shape[i]) return false;
    }
    return true;
  }

  @override
  String toString() {
    return 'Tensor(shape: $shape, data: ${data.take(10)}${data.length > 10 ? '...' : ''})';
  }
}