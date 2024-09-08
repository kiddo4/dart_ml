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

  Tensor operator +(Tensor other) =>
      _elementwiseBinaryOp(other, (a, b) => a + b);
  Tensor operator -(Tensor other) =>
      _elementwiseBinaryOp(other, (a, b) => a - b);
  Tensor operator *(Tensor other) =>
      _elementwiseBinaryOp(other, (a, b) => a * b);
  Tensor operator /(Tensor other) =>
      _elementwiseBinaryOp(other, (a, b) => a / b);

  Tensor _elementwiseBinaryOp(
      Tensor other, double Function(double, double) operation) {
    if (!_broadcastCompatible(shape, other.shape)) {
      throw ArgumentError('Tensor shapes are not compatible for broadcasting');
    }

    final broadcastShape = _broadcastShape(shape, other.shape);
    final broadcastedThis = broadcastTo(broadcastShape);
    final broadcastedOther = other.broadcastTo(broadcastShape);

    final newData = Float32List(broadcastedThis.data.length);
    for (var i = 0; i < newData.length; i++) {
      newData[i] = operation(broadcastedThis.data[i], broadcastedOther.data[i]);
    }

    return Tensor(broadcastShape, newData);
  }

// Helper function to check if two shapes are compatible for broadcasting
  bool _broadcastCompatible(List<int> shapeA, List<int> shapeB) {
    final lenA = shapeA.length;
    final lenB = shapeB.length;
    final maxLen = math.max(lenA, lenB);

    for (int i = 0; i < maxLen; i++) {
      final dimA = (i < lenA) ? shapeA[lenA - 1 - i] : 1;
      final dimB = (i < lenB) ? shapeB[lenB - 1 - i] : 1;
      if (dimA != dimB && dimA != 1 && dimB != 1) {
        return false;
      }
    }
    return true;
  }

// Helper function to compute the broadcast shape
  List<int> _broadcastShape(List<int> shapeA, List<int> shapeB) {
    final lenA = shapeA.length;
    final lenB = shapeB.length;
    final maxLen = math.max(lenA, lenB);

    final result = List<int>.filled(maxLen, 0);
    for (int i = 0; i < maxLen; i++) {
      final dimA = (i < lenA) ? shapeA[lenA - 1 - i] : 1;
      final dimB = (i < lenB) ? shapeB[lenB - 1 - i] : 1;
      result[maxLen - 1 - i] = math.max(dimA, dimB);
    }
    return result;
  }

  Tensor elementwiseOperation(double Function(double) operation) {
    final newData = Float32List(data.length);
    for (var i = 0; i < data.length; i++) {
      newData[i] = operation(data[i]);
    }
    return Tensor(shape, newData);
  }

  Tensor exp() => elementwiseOperation(math.exp);
  Tensor log() => elementwiseOperation(math.log);
  Tensor abs() => elementwiseOperation((x) => x.abs());
  Tensor sqrt() => elementwiseOperation(math.sqrt);

  Tensor relu() => elementwiseOperation((x) => math.max(0, x));
  Tensor sigmoid() => elementwiseOperation((x) => 1 / (1 + math.exp(-x)));
  Tensor tanh() => elementwiseOperation((x) => math.sin(x) / math.cos(x));

  double mse(Tensor other) {
    if (!_shapeMatch(other)) {
      throw ArgumentError('Tensor shapes do not match for MSE calculation');
    }
    double sum = 0;
    for (var i = 0; i < data.length; i++) {
      final diff = data[i] - other.data[i];
      sum += diff * diff;
    }
    return sum / data.length;
  }

  double crossEntropy(Tensor other) {
    if (!_shapeMatch(other)) {
      throw ArgumentError(
          'Tensor shapes do not match for cross-entropy calculation');
    }
    double sum = 0;
    for (var i = 0; i < data.length; i++) {
      sum += other.data[i] * math.log(data[i]);
    }
    return -sum;
  }

  Tensor sum({int? axis}) {
    if (axis == null) {
      final sum = data.reduce((a, b) => a + b);
      return Tensor([1], Float32List.fromList([sum]));
    } else {
      if (shape.length != 2) {
        throw UnimplementedError(
            'Sum along axis is only implemented for 2D tensors');
      }
      final newShape = List<int>.from(shape);
      newShape[axis] = 1;
      final newData = Float32List(newShape.reduce((a, b) => a * b));
      if (axis == 0) {
        for (var i = 0; i < shape[1]; i++) {
          double sum = 0;
          for (var j = 0; j < shape[0]; j++) {
            sum += data[j * shape[1] + i];
          }
          newData[i] = sum;
        }
      } else {
        for (var i = 0; i < shape[0]; i++) {
          double sum = 0;
          for (var j = 0; j < shape[1]; j++) {
            sum += data[i * shape[1] + j];
          }
          newData[i] = sum;
        }
      }
      return Tensor(newShape, newData);
    }
  }

  Tensor mean({int? axis}) {
    final sum = this.sum(axis: axis);
    final divisor = axis == null ? data.length : shape[axis];
    return sum.elementwiseOperation((x) => x / divisor);
  }

  Tensor matmul(Tensor other) {
    if (shape.length != 2 ||
        other.shape.length != 2 ||
        shape[1] != other.shape[0]) {
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

  Tensor transpose() {
    if (shape.length != 2) {
      throw UnimplementedError('Transpose is only implemented for 2D tensors');
    }
    final newShape = [shape[1], shape[0]];
    final newData = Float32List(newShape.reduce((a, b) => a * b));
    for (var i = 0; i < shape[0]; i++) {
      for (var j = 0; j < shape[1]; j++) {
        newData[j * shape[0] + i] = data[i * shape[1] + j];
      }
    }
    return Tensor(newShape, newData);
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

extension BroadcastTensor on Tensor {
  Tensor broadcastTo(List<int> newShape) {
    if (shape.length > newShape.length) {
      throw ArgumentError('Cannot broadcast to a shape with fewer dimensions');
    }

    List<int> paddedShape =
        List.filled(newShape.length - shape.length, 1) + shape;
    List<int> repeats =
        List.generate(newShape.length, (i) => newShape[i] ~/ paddedShape[i]);

    Float32List newData = Float32List(newShape.reduce((a, b) => a * b));
    int newSize = newData.length;
    int oldSize = data.length;

    for (int i = 0; i < newSize; i++) {
      List<int> newIndices = _unravel(i, newShape);
      List<int> oldIndices = List.generate(
          paddedShape.length, (j) => newIndices[j] % paddedShape[j]);
      int oldIndex = _ravel(oldIndices, paddedShape);
      newData[i] = data[oldIndex];
    }

    return Tensor(newShape, newData);
  }

  List<int> _unravel(int index, List<int> shape) {
    List<int> indices = List.filled(shape.length, 0);
    for (int i = shape.length - 1; i >= 0; i--) {
      indices[i] = index % shape[i];
      index ~/= shape[i];
    }
    return indices;
  }

  int _ravel(List<int> indices, List<int> shape) {
    int index = 0;
    int multiplier = 1;
    for (int i = shape.length - 1; i >= 0; i--) {
      index += indices[i] * multiplier;
      multiplier *= shape[i];
    }
    return index;
  }
}
