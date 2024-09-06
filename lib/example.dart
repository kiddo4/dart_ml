import 'dart:typed_data';

import 'package:dart_ml/core/tensors.dart';

import 'core/autograd.dart';

void main() {
  var a = Variable(Tensor([2, 2], Float32List.fromList([1, 2, 3, 4])));
  var b = Variable(Tensor([2, 2], Float32List.fromList([2, 3, 4, 5])));
  var c = Variable(Tensor([2, 2], Float32List.fromList([3, 4, 5, 6])));

  var d = add(a, b);
  var e = multiply(d, c);

  e.backward();

  print('Gradient of a:');
  print(a.grad!.data);

  print('Gradient of b:');
  print(b.grad!.data);

  print('Gradient of c:');
  print(c.grad!.data);
}