import 'dart:convert';
import 'dart:io';

import 'package:csv/csv.dart';

class DataLoader {

  Future<List<Map<String, String>>> loadCSV(String filePath) async {
  try {
    final file = File(filePath);
    final content = await file.readAsString();

    // Parse CSV content
    final csvConverter = CsvToListConverter(eol: '\n');
    final List<List<dynamic>> rows = csvConverter.convert(content);

    if (rows.isEmpty) {
      throw Exception("The file is empty.");
    }

    final headers = rows.first.map((header) => header.toString()).toList();

    // Ensure that all rows have the same number of fields as the header
    return rows.skip(1).where((row) {
      if (row.length != headers.length) {
        print("Warning: Skipping invalid row: $row");
        return false;
      }
      return true;
    }).map((row) {
      return Map<String, String>.fromIterables(headers, row.map((e) => e.toString()));
    }).toList();
  } catch (e) {
    print('Error loading CSV file: $e');
    rethrow;
  }
}


  Future<List<Map<String, dynamic>>> loadJson(String filePath) async {
  try {
    final file = File(filePath);
    final content = await file.readAsString();
    final decoded = json.decode(content);

    if (decoded is List) {
      return List<Map<String, dynamic>>.from(decoded);
    } else {
      throw FormatException('JSON data is not a list.');
    }
  } catch (e) {
    print('Error loading JSON file: $e');
    rethrow;
  }
}

}
