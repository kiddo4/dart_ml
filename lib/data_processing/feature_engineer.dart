import 'dart:math';

class FeatureEngineer {
  /// Normalizes a feature using Min-Max Scaling.
  ///
  /// Returns a list of maps with the specified column normalized to the range [0, 1].
  /// If the column has no valid numerical values, a warning is printed, and the original data is returned.
  List<Map<String, dynamic>> normalize(
    List<Map<String, dynamic>> data,
    String column,
  ) {
    final values = data
        .map((row) => _parseDouble(row[column]))
        .where((value) => value != null)
        .cast<double>()
        .toList();

    if (values.isEmpty) {
      print('Warning: No valid numerical values found in column $column');
      return data;
    }

    final minValue = values.reduce(min);
    final maxValue = values.reduce(max);
    final range = maxValue - minValue;

    return data.map((row) {
      final value = _parseDouble(row[column]);
      if (value != null && row[column] is! String) {
        row[column] = range != 0 ? (value - minValue) / range : 0.0;
      }
      return row;
    }).toList();
  }

  /// Standardizes a feature using Z-Score Scaling.
  ///
  /// Returns a list of maps with the specified column standardized to have a mean of 0 and a standard deviation of 1.
  /// If the column has no valid numerical values, a warning is printed, and the original data is returned.
  List<Map<String, dynamic>> standardize(
    List<Map<String, dynamic>> data,
    String column,
  ) {
    final values = data
        .map((row) => _parseDouble(row[column]))
        .where((value) => value != null)
        .cast<double>()
        .toList();

    if (values.isEmpty) {
      print('Warning: No valid numerical values found in column $column');
      return data;
    }

    final meanValue = values.reduce((a, b) => a + b) / values.length;
    final variance = values.map((v) => pow(v - meanValue, 2)).reduce((a, b) => a + b) / values.length;
    final stdDev = sqrt(variance);

    return data.map((row) {
      final value = _parseDouble(row[column]);
      if (value != null && row[column] is! String) {
        row[column] = stdDev != 0 ? (value - meanValue) / stdDev : 0.0;
      }
      return row;
    }).toList();
  }

  /// Parses a value to a double, handling various types.
  double? _parseDouble(dynamic value) {
    if (value == null) return null;
    if (value is num) return value.toDouble();
    if (value is String) return double.tryParse(value);
    return null;
  }
}
