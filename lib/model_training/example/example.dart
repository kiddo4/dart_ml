import 'dart:io';

import 'package:dart_ml/model_training/ml_models.dart';
import 'package:image/image.dart' as img;


void main() async {
  final loader = MNISTLoader();
  final trainImages = loader.loadImages('assets/train-images.idx3-ubyte');
  final trainLabels = loader.loadLabels('assets/train-labels.idx1-ubyte');
  final testImages = loader.loadImages('assets/t10k-images.idx3-ubyte');
  final testLabels = loader.loadLabels('assets/t10k-labels.idx1-ubyte');
  
  final model = MNISTModel();
  
  // Train the model
  trainMNISTModel(model, trainImages, trainLabels, testImages, testLabels, 50, 0.001);
  
  // Save the trained model (implement model saving/loading)
  // saveModel(model, 'mnist_model.json');
  
  // Example usage: recognize a digit from a file
  final image = img.decodeImage(File('sample_digit.png').readAsBytesSync())!;
  final prediction = recognizeDigit(model, image);
  print('Predicted digit: $prediction');
}
