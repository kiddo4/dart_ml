import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:dart_ml/core/tensors.dart';

void main() {
  group('Tensor', () {
    test('Creation and shape validation', () {
      final tensor = Tensor([2, 3], Float32List.fromList([1, 2, 3, 4, 5, 6]));
      expect(tensor.shape, [2, 3]);
      expect(tensor.data.length, 6);
    });

    test('Zeros and Ones', () {
      final zeros = Tensor.zeros([2, 2]);
      final ones = Tensor.ones([2, 2]);
      expect(zeros.data.every((element) => element == 0), isTrue);
      expect(ones.data.every((element) => element == 1), isTrue);
    });

    test('Random normal distribution', () {
      final tensor = Tensor.randn([2, 3]);
      expect(tensor.shape, [2, 3]);
      expect(tensor.data.length, 6);
    });

    test('Addition', () {
      final t1 = Tensor([2, 2], Float32List.fromList([1, 2, 3, 4]));
      final t2 = Tensor([2, 2], Float32List.fromList([4, 3, 2, 1]));
      final result = t1 + t2;
      expect(result.data, [5, 5, 5, 5]);
    });

    test('Element-wise operations', () {
      final t1 = Tensor([2, 2], Float32List.fromList([1, 4, 9, 16]));
      final sqrtResult = t1.sqrt();
      expect(sqrtResult.data, [1, 2, 3, 4]);
    });

    test('Matrix multiplication', () {
      final t1 = Tensor([2, 3], Float32List.fromList([1, 2, 3, 4, 5, 6]));
      final t2 = Tensor([3, 2], Float32List.fromList([7, 8, 9, 10, 11, 12]));
      final result = t1.matmul(t2);
      expect(result.shape, [2, 2]);
      expect(result.data, [58, 64, 139, 154]);
    });

    test('Mean calculation', () {
      final t1 = Tensor([2, 2], Float32List.fromList([2, 4, 6, 8]));
      final mean = t1.mean();
      expect(mean.data.first, 5.0);
    });
  });

  

}
