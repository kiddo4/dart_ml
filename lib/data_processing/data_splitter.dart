import 'dart:math';

class DataSplitter {
  // Existing trainTestSplit method
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

  // New method for train-validation-test split
  Map<String, List<Map<String, dynamic>>> trainValidationTestSplit(
    List<Map<String, dynamic>> data, {
    required double validationSize,
    required double testSize,
    bool shuffle = true,
  }) {
    // Validate sizes
    if (validationSize < 0.0 || validationSize > 1.0 || testSize < 0.0 || testSize > 1.0) {
      throw ArgumentError('validationSize and testSize must be between 0 and 1');
    }
    if (validationSize + testSize >= 1.0) {
      throw ArgumentError('Sum of validationSize and testSize must be less than 1');
    }

    // Shuffle the data if required
    final dataToSplit = List<Map<String, dynamic>>.from(data);
    if (shuffle) {
      dataToSplit.shuffle(Random());
    }

    // Calculate split indices
    final totalCount = dataToSplit.length;
    final testCount = (totalCount * testSize).round();
    final validationCount = (totalCount * validationSize).round();
    final trainCount = totalCount - testCount - validationCount;

    return {
      'train': dataToSplit.take(trainCount).toList(),
      'validation': dataToSplit.skip(trainCount).take(validationCount).toList(),
      'test': dataToSplit.skip(trainCount + validationCount).toList(),
    };
  }
}