import 'dart:math';

class DataSplitter {
  // Split the data into training and test sets
  Map<String, List<Map<String, dynamic>>> trainTestSplit(
    List<Map<String, dynamic>> data, double testSize) {
  final testCount = (data.length * testSize).round();
  final trainCount = data.length - testCount;

  return {
    'train': data.take(trainCount).toList(),
    'test': data.skip(trainCount).toList(),
  };
}

}
