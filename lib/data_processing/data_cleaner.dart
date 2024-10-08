import 'dart:math' as math;

class DataCleaner {
  // Handles missing data by filling with a specified value
  List<Map<String, dynamic>> fillMissingValues(
      List<Map<String, dynamic>> data, String column, dynamic fillValue) {
    return data.map((row) {
      if (row[column] == null) {
        row = Map<String, dynamic>.from(
            row); // Create a new map to avoid modifying the original
        row[column] = fillValue;
      }
      return row;
    }).toList();
  }

  // Drop rows with missing data
  List<Map<String, dynamic>> dropMissingRows(
    List<Map<String, dynamic>> data,
    String column,
  ) {
    return data
        .where((row) =>
            row[column] != null &&
            (row[column] is! String || (row[column] as String).isNotEmpty))
        .toList();
  }

  // Convert categorical features to numerical encoding
  List<Map<String, dynamic>> encodeCategorical(
    List<Map<String, dynamic>> data,
    String column,
    Map<String, int> mapping,
  ) {
    return data.map((row) {
      if (mapping.containsKey(row[column])) {
        row = Map<String, dynamic>.from(
            row); // Create a new map to avoid modifying the original
        row[column] = mapping[row[column]]; // Keep as int
      }
      return row;
    }).toList();
  }

  // Normalize numerical features
  List<Map<String, dynamic>> normalizeNumerical(
    List<Map<String, dynamic>> data,
    String column,
  ) {
    var values = data.map((row) => row[column]).whereType<num>().toList();

    if (values.isEmpty) return List.from(data);

    var min = values.reduce(math.min);
    var max = values.reduce(math.max);
    var range = max - min;

    return data.map((row) {
      var newRow = Map<String, dynamic>.from(row);
      if (newRow[column] is num) {
        newRow[column] = range != 0 ? (newRow[column] - min) / range : 0;
      }
      return newRow;
    }).toList();
  }

  // Handle outliers using IQR method
  List<Map<String, dynamic>> handleOutliers(
      List<Map<String, dynamic>> data, String column,
      {bool remove = false}) {
    var values = data.map((row) => row[column] as num).toList()..sort();
    var q1 = values[values.length ~/ 4];
    var q3 = values[(values.length * 3) ~/ 4];
    var iqr = q3 - q1;
    var lowerBound = q1 - 1.5 * iqr;
    var upperBound = q3 + 1.5 * iqr;

    return data.where((row) {
      if (row[column] is num) {
        var value = row[column] as num;
        if (value < lowerBound || value > upperBound) {
          return !remove;
        }
      }
      return true;
    }).toList();
  }
}
