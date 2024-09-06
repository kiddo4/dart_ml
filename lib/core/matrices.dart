import 'dart:typed_data';
import 'dart:math' as math;

class Matrix {
  final int rows;
  final int cols;
  final Float32List data;

  Matrix(this.rows, this.cols, this.data) {
    if (data.length != rows * cols) {
      throw ArgumentError('Data length does not match matrix dimensions.');
    }
  }

  factory Matrix.zeros(int rows, int cols) {
    return Matrix(rows, cols, Float32List(rows * cols));
  }

  factory Matrix.ones(int rows, int cols) {
    return Matrix(rows, cols, Float32List(rows * cols)..fillRange(0, rows * cols, 1.0));
  }

  factory Matrix.randn(int rows, int cols) {
    final size = rows * cols;
    final data = Float32List(size);
    final random = math.Random();
    for (var i = 0; i < size; i++) {
      data[i] = _boxMullerTransform(random);
    }
    return Matrix(rows, cols, data);
  }

  static double _boxMullerTransform(math.Random random) {
    final u1 = 1.0 - random.nextDouble();
    final u2 = 1.0 - random.nextDouble();
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2);
  }

  Matrix operator +(Matrix other) {
    if (rows != other.rows || cols != other.cols) {
      throw ArgumentError('Matrix dimensions do not match for addition.');
    }
    final newData = Float32List(data.length);
    for (var i = 0; i < data.length; i++) {
      newData[i] = data[i] + other.data[i];
    }
    return Matrix(rows, cols, newData);
  }

  Matrix operator -(Matrix other) {
    if (rows != other.rows || cols != other.cols) {
      throw ArgumentError('Matrix dimensions do not match for subtraction.');
    }
    final newData = Float32List(data.length);
    for (var i = 0; i < data.length; i++) {
      newData[i] = data[i] - other.data[i];
    }
    return Matrix(rows, cols, newData);
  }

  Matrix operator *(Matrix other) {
    if (rows != other.rows || cols != other.cols) {
      throw ArgumentError('Matrix dimensions do not match for element-wise multiplication.');
    }
    final newData = Float32List(data.length);
    for (var i = 0; i < data.length; i++) {
      newData[i] = data[i] * other.data[i];
    }
    return Matrix(rows, cols, newData);
  }

  Matrix transpose() {
    final newData = Float32List(rows * cols);
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        newData[j * rows + i] = data[i * cols + j];
      }
    }
    return Matrix(cols, rows, newData);
  }

  Matrix matmul(Matrix other) {
    if (cols != other.rows) {
      throw ArgumentError('Invalid dimensions for matrix multiplication.');
    }
    final newData = Float32List(rows * other.cols);
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < other.cols; j++) {
        double sum = 0.0;
        for (var k = 0; k < cols; k++) {
          sum += data[i * cols + k] * other.data[k * other.cols + j];
        }
        newData[i * other.cols + j] = sum;
      }
    }
    return Matrix(rows, other.cols, newData);
  }

  Matrix map(double Function(double) func) {
    final newData = Float32List(data.length);
    for (var i = 0; i < data.length; i++) {
      newData[i] = func(data[i]);
    }
    return Matrix(rows, cols, newData);
  }

  @override
  String toString() {
    return 'Matrix(rows: $rows, cols: $cols, data: ${data.take(10)}${data.length > 10 ? '...' : ''})';
  }
}
