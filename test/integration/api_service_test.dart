// import 'dart:convert';
// import 'dart:io';

// import 'package:dart_ml/integration/api_integration.dart';
// import 'package:dio/dio.dart';
// import 'package:mockito/mockito.dart';
// import 'package:test/test.dart';


// // Mock class
// class MockDio extends Mock implements Dio {}

// void main() {
//   group('ApiService Tests', () {
//     late ApiService apiService;
//     late MockDio mockDio;

//     setUp(() {
//       mockDio = MockDio();
//       apiService = ApiService.withDio(mockDio, 'token');
//     });

//     test('fetchData returns data on success', () async {
//       when(mockDio.get('any', options: anyNamed('options')))
//           .thenAnswer(
//         (_) async => Response(
//           data: {'key': 'value'},
//           statusCode: 200,
//           requestOptions: RequestOptions(path: ''),
//         ),
//       );

//       final data = await apiService.fetchData('/endpoint');
//       expect(data, {'key': 'value'});
//     });

//     test('fetchData retries on failure', () async {
//       when(mockDio.get('', options: anyNamed('options')))
//           .thenThrow(DioError(
//         requestOptions: RequestOptions(path: ''),
//       ));

//       // Expect that fetchData throws an error after retries
//       expect(
//         () async => await apiService.fetchData('/endpoint'),
//         throwsA(isA<DioError>()),
//       );
//     });

//     test('sendResults returns true on success', () async {
//       when(mockDio.post('any', data: anyNamed('data'), options: anyNamed('options')))
//           .thenAnswer(
//         (_) async => Response(
//           statusCode: 200,
//           requestOptions: RequestOptions(path: ''),
//         ),
//       );

//       final success = await apiService.sendResults('/endpoint', {'result': 'data'});
//       expect(success, true);
//     });

//     test('sendResults returns false on failure', () async {
//       when(mockDio.post('any', data: anyNamed('data'), options: anyNamed('options')))
//           .thenThrow(DioError(
//         requestOptions: RequestOptions(path: ''),
//       ));

//       final success = await apiService.sendResults('/endpoint', {'result': 'data'});
//       expect(success, false);
//     });

//     test('downloadFile downloads file successfully', () async {
//       final filePath = 'test_download.txt';

//       when(mockDio.get('any', options: anyNamed('options')))
//           .thenAnswer(
//         (_) async => Response(
//           data: Stream.fromIterable([utf8.encode('test data')]),
//           statusCode: 200,
//           requestOptions: RequestOptions(path: ''),
//         ),
//       );

//       await apiService.downloadFile('/endpoint', filePath);
//       final file = File(filePath);
//       expect(await file.readAsString(), 'test data');
//       file.deleteSync(); // Clean up test file
//     });

//     test('uploadFile uploads file successfully', () async {
//       final filePath = 'test_upload.txt';
//       final file = File(filePath);
//       file.writeAsStringSync('test data');

//       when(mockDio.post('any', data: anyNamed('data'), options: anyNamed('options')))
//           .thenAnswer(
//         (_) async => Response(
//           statusCode: 200,
//           requestOptions: RequestOptions(path: ''),
//         ),
//       );

//       await apiService.uploadFile('/endpoint', filePath);
//       file.deleteSync(); // Clean up test file
//     });

//     test('uploadFile handles upload failure', () async {
//       final filePath = 'test_upload.txt';
//       final file = File(filePath);
//       file.writeAsStringSync('test data');

//       when(mockDio.post('any', data: anyNamed('data'), options: anyNamed('options')))
//           .thenThrow(DioError(
//         requestOptions: RequestOptions(path: ''),
//       ));

//       await apiService.uploadFile('/endpoint', filePath);
//       file.deleteSync(); // Clean up test file
//     });
//   });
// }
