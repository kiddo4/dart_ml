import 'dart:math';

class DataSplitter {
  // Split the data into training and test sets with optional shuffling
  Map<String, List<Map<String, dynamic>>> trainTestSplit(
    List<Map<String, dynamic>> data, {
    required double testSize,
    bool shuffle = false,
  }) {
    // Validate testSize
    if (testSize < 0.0 || testSize > 1.0) {
      throw ArgumentError('testSize must be between 0 and 1');
    }

    // Shuffle the data if required
    final dataToSplit = List<Map<String, dynamic>>.from(data);
    if (shuffle) {
      dataToSplit.shuffle(Random());
    }

    // Calculate split indices
    final testCount = (dataToSplit.length * testSize).round();
    final trainCount = dataToSplit.length - testCount;

    return {
      'train': dataToSplit.take(trainCount).toList(),
      'test': dataToSplit.skip(trainCount).toList(),
    };
  }
}
