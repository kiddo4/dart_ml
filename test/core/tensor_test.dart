import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_ml/core/tensors.dart';

void main() {
  group('Tensor', () {
    test('creation', () {
      final t = Tensor([2, 3], Float32List.fromList([1, 2, 3, 4, 5, 6]));
      expect(t.shape, equals([2, 3]));
      expect(t.data, equals(Float32List.fromList([1, 2, 3, 4, 5, 6])));
    });

    test('zeros', () {
      final t = Tensor.zeros([2, 3]);
      expect(t.shape, equals([2, 3]));
      expect(t.data, equals(Float32List(6)..fillRange(0, 6, 0.0)));
    });

    test('ones', () {
      final t = Tensor.ones([2, 2]);
      expect(t.shape, equals([2, 2]));
      expect(t.data, equals(Float32List(4)..fillRange(0, 4, 1.0)));
    });

    test('randn', () {
      final t = Tensor.randn([2, 3]);
      expect(t.shape, equals([2, 3]));
      expect(t.data.length, equals(6));
      // Check if values are within a reasonable range for a normal distribution
      for (var value in t.data) {
        expect(value, inInclusiveRange(-10, 10));
      }
    });

    test('reshape', () {
      final t = Tensor([2, 3], Float32List.fromList([1, 2, 3, 4, 5, 6]));
      final reshaped = t.reshape([3, 2]);
      expect(reshaped.shape, equals([3, 2]));
      expect(reshaped.data, equals(t.data));
    });

    test('addition', () {
      final t1 = Tensor([2, 2], Float32List.fromList([1, 2, 3, 4]));
      final t2 = Tensor([2, 2], Float32List.fromList([5, 6, 7, 8]));
      final result = t1 + t2;
      expect(result.shape, equals([2, 2]));
      expect(result.data, equals(Float32List.fromList([6, 8, 10, 12])));
    });

    test('subtraction', () {
      final t1 = Tensor([2, 2], Float32List.fromList([5, 6, 7, 8]));
      final t2 = Tensor([2, 2], Float32List.fromList([1, 2, 3, 4]));
      final result = t1 - t2;
      expect(result.shape, equals([2, 2]));
      expect(result.data, equals(Float32List.fromList([4, 4, 4, 4])));
    });

    test('element-wise multiplication', () {
      final t1 = Tensor([2, 2], Float32List.fromList([1, 2, 3, 4]));
      final t2 = Tensor([2, 2], Float32List.fromList([5, 6, 7, 8]));
      final result = t1 * t2;
      expect(result.shape, equals([2, 2]));
      expect(result.data, equals(Float32List.fromList([5, 12, 21, 32])));
    });

    test('matrix multiplication', () {
      final t1 = Tensor([2, 3], Float32List.fromList([1, 2, 3, 4, 5, 6]));
      final t2 = Tensor([3, 2], Float32List.fromList([7, 8, 9, 10, 11, 12]));
      final result = t1.matmul(t2);
      expect(result.shape, equals([2, 2]));
      expect(result.data, equals(Float32List.fromList([58, 64, 139, 154])));
    });

    test('invalid shape for creation', () {
      expect(() => Tensor([2, 3], Float32List(5)), throwsArgumentError);
    });

    test('invalid shape for addition', () {
      final t1 = Tensor([2, 2], Float32List(4));
      final t2 = Tensor([2, 3], Float32List(6));
      expect(() => t1 + t2, throwsArgumentError);
    });

    test('invalid shape for matrix multiplication', () {
      final t1 = Tensor([2, 3], Float32List(6));
      final t2 = Tensor([2, 2], Float32List(4));
      expect(() => t1.matmul(t2), throwsArgumentError);
    });
  });
}