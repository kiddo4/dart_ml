import 'dart:math';

class FeatureEngineer {
  // Normalize a feature (Min-Max Scaling)
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
      if (value != null && row[column] is! String) {  // Ensure correct type handling
        row[column] = range != 0 ? (value - minValue) / range : 0.0;
      }
      return row;
    }).toList();
  }

  // Standardize a feature (Z-Score Scaling)
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
      if (value != null && row[column] is! String) {  // Ensure correct type handling
        row[column] = stdDev != 0 ? (value - meanValue) / stdDev : 0.0;
      }
      return row;
    }).toList();
  }

  // Helper method to safely parse double values
  double? _parseDouble(dynamic value) {
    if (value == null) return null;
    if (value is num) return value.toDouble();
    if (value is String) return double.tryParse(value);
    return null;
  }
}
