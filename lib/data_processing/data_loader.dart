import 'dart:convert';
import 'dart:io';

import 'package:csv/csv.dart';

class DataLoader {

  Future<List<Map<String, String>>> loadCSV(String filePath) async {
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
  }


  Future<List<Map<String, dynamic>>> loadJson(String filePath) async {
    final file = File(filePath);
    final content = await file.readAsString();
    return List<Map<String, dynamic>>.from(json.decode(content));
  }
}
