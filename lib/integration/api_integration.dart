import 'dart:convert';
import 'dart:io';
import 'package:dio/dio.dart';

class ApiService {
  final Dio _dio;
  final String _token;

  ApiService(String baseUrl, [this._token = ''])
      : _dio = Dio(BaseOptions(
          baseUrl: baseUrl,
        ));

  ApiService.withDio(Dio dio, [this._token = '']) : _dio = dio;

  Future<Map<String, dynamic>> fetchData(String endpoint, {int retries = 3}) async {
    for (int i = 0; i < retries; i++) {
      try {
        final response = await _dio.get(
          endpoint,
          options: Options(
            headers: _token.isNotEmpty
                ? {'Authorization': 'Bearer $_token'}
                : {},
          ),
        );
        return response.data as Map<String, dynamic>;
      } catch (e) {
        print('Error fetching data: $e');
        if (i == retries - 1) rethrow;
        await Future.delayed(Duration(seconds: 2 * (i + 1))); // Exponential backoff
      }
    }
    return {};
  }

  Future<bool> sendResults(String endpoint, Map<String, dynamic> results) async {
    try {
      final response = await _dio.post(
        endpoint,
        data: json.encode(results),
        options: Options(
          headers: {
            'Content-Type': 'application/json',
            if (_token.isNotEmpty) 'Authorization': 'Bearer $_token',
          },
        ),
      );
      return response.statusCode == 200;
    } catch (e) {
      print('Error sending results: $e');
      return false;
    }
  }

  Future<void> downloadFile(String endpoint, String savePath) async {
    try {
      final response = await _dio.get(
        endpoint,
        options: Options(responseType: ResponseType.stream),
      );
      final file = File(savePath);
      final raf = file.openSync(mode: FileMode.write);
      await response.data.stream.pipe(raf);
      raf.closeSync();
      print('File downloaded successfully to $savePath');
    } catch (e) {
      print('Error downloading file: $e');
    }
  }

  Future<void> uploadFile(String endpoint, String filePath) async {
    try {
      final file = File(filePath);
      final formData = FormData.fromMap({
        'file': await MultipartFile.fromFile(filePath, filename: file.uri.pathSegments.last),
      });
      final response = await _dio.post(endpoint, data: formData);
      if (response.statusCode == 200) {
        print('File uploaded successfully');
      } else {
        print('Failed to upload file: ${response.statusCode}');
      }
    } catch (e) {
      print('Error uploading file: $e');
    }
  }
}
