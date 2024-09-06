import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_ml/core/matrices.dart';

void main() {
  group('Matrix', () {
    test('Creation and shape validation', () {
      final matrix = Matrix(2, 3, Float32List.fromList([1, 2, 3, 4, 5, 6]));
      expect(matrix.rows, 2);
      expect(matrix.cols, 3);
      expect(matrix.data.length, 6);
    });

    test('Zeros and Ones', () {
      final zeros = Matrix.zeros(2, 2);
      final ones = Matrix.ones(2, 2);
      expect(zeros.data.every((element) => element == 0), isTrue);
      expect(ones.data.every((element) => element == 1), isTrue);
    });

    test('Random normal distribution', () {
      final matrix = Matrix.randn(2, 3);
      expect(matrix.rows, 2);
      expect(matrix.cols, 3);
      expect(matrix.data.length, 6);
    });

    test('Addition', () {
      final m1 = Matrix(2, 2, Float32List.fromList([1, 2, 3, 4]));
      final m2 = Matrix(2, 2, Float32List.fromList([4, 3, 2, 1]));
      final result = m1 + m2;
      expect(result.data, [5, 5, 5, 5]);
    });

    test('Transpose', () {
      final matrix = Matrix(2, 3, Float32List.fromList([1, 2, 3, 4, 5, 6]));
      final transposed = matrix.transpose();
      expect(transposed.rows, 3);
      expect(transposed.cols, 2);
      expect(transposed.data, [1, 4, 2, 5, 3, 6]);
    });

    test('Matrix multiplication', () {
      final m1 = Matrix(2, 3, Float32List.fromList([1, 2, 3, 4, 5, 6]));
      final m2 = Matrix(3, 2, Float32List.fromList([7, 8, 9, 10, 11, 12]));
      final result = m1.matmul(m2);
      expect(result.rows, 2);
      expect(result.cols, 2);
      expect(result.data, [58, 64, 139, 154]);
    });
  });
}
