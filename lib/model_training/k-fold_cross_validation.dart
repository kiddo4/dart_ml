import 'dart:math';

import 'package:ml_dart/ml_dart.dart';

List<Map<String, Map<String, List<Tensor>>>> kFoldCrossValidation(List<Tensor> data, List<Tensor> labels, int k) {
  assert(data.length == labels.length, "Data and labels must have the same length");
  assert(k > 1, "k must be greater than 1");

  List<Map<String, Map<String, List<Tensor>>>> folds = [];
  int foldSize = data.length ~/ k;
  List<int> indices = List<int>.generate(data.length, (i) => i);
  indices.shuffle(Random());

  for (int i = 0; i < k; i++) {
    int startIdx = i * foldSize;
    int endIdx = (i == k - 1) ? data.length : (i + 1) * foldSize;

    List<int> testIndices = indices.sublist(startIdx, endIdx);
    List<int> trainIndices = indices.where((idx) => !testIndices.contains(idx)).toList();

    folds.add({
      'train': {
        'data': trainIndices.map((idx) => data[idx]).toList(),
        'labels': trainIndices.map((idx) => labels[idx]).toList(),
      },
      'test': {
        'data': testIndices.map((idx) => data[idx]).toList(),
        'labels': testIndices.map((idx) => labels[idx]).toList(),
      },
    });
  }

  return folds;
}