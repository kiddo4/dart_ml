import 'dart:math';
import 'package:ml_dart/ml_dart.dart';

class LogisticRegression implements Model {
  final LinearLayer linearLayer;
  final Sigmoid sigmoid;

  LogisticRegression(int inputSize, int outputSize)
      : linearLayer = LinearLayer(inputSize, outputSize),
        sigmoid = Sigmoid();

  @override
  Tensor forward(Tensor input) {
    Tensor linearOutput = linearLayer.forward(input);
    return sigmoid.forward(linearOutput);
  }

  @override
  void backward(Tensor gradOutput) {
    sigmoid.backward(gradOutput);
    linearLayer.backward(gradOutput);
  }

  @override
  void updateParameters(double learningRate) {
    linearLayer.updateParameters(learningRate);
  }

  void train(List<Tensor> trainData, List<Tensor> trainLabels,
      List<Tensor> valData, List<Tensor> valLabels,
      {required int epochs, required double learningRate, int patience = 10}) {
    double bestValLoss = double.infinity;
    int patienceCounter = 0;

    for (int epoch = 0; epoch < epochs; epoch++) {
      double totalLoss = 0.0;
      int correct = 0;

      for (int i = 0; i < trainData.length; i++) {
        Tensor output = forward(trainData[i]);
        Tensor loss = crossEntropyLoss(output, trainLabels[i]);
        totalLoss += loss.sum().data[0];

        if (argmax(output) == argmax(trainLabels[i])) correct++;

        Tensor gradOutput = output - trainLabels[i];
        backward(gradOutput);
        updateParameters(learningRate);
      }

      double accuracy = correct / trainData.length;
      print("Epoch $epoch: Loss: ${totalLoss / trainData.length}, Accuracy: $accuracy");

      // Validation
      double valLoss = 0.0;
      int valCorrect = 0;

      for (int i = 0; i < valData.length; i++) {
        Tensor output = forward(valData[i]);
        Tensor loss = crossEntropyLoss(output, valLabels[i]);
        valLoss += loss.sum().data[0];

        if (argmax(output) == argmax(valLabels[i])) valCorrect++;
      }

      double valAccuracy = valCorrect / valData.length;
      print("Validation Loss: ${valLoss / valData.length}, Accuracy: $valAccuracy");

      // Early stopping
      if (valLoss < bestValLoss) {
        bestValLoss = valLoss;
        patienceCounter = 0;
      } else {
        patienceCounter++;
        if (patienceCounter >= patience) {
          print("Early stopping at epoch $epoch");
          break;
        }
      }
    }
  }

  double evaluate(List<Tensor> testImages, List<Tensor> testLabels) {
    int correct = 0;

    for (int i = 0; i < testImages.length; i++) {
      Tensor output = forward(testImages[i]);
      if (argmax(output) == argmax(testLabels[i])) correct++;
    }

    return correct / testImages.length;
  }

  List<double> featureImportance() {
    List<double> importance =
        List<double>.filled(linearLayer.weights.shape[0], 0.0);

    for (int i = 0; i < linearLayer.weights.shape[0]; i++) {
      for (int j = 0; j < linearLayer.weights.shape[1]; j++) {
        importance[i] += linearLayer
            .weights.data[i * linearLayer.weights.shape[1] + j]
            .abs();
      }
    }

    // Normalize importance scores
    double maxImportance = importance.reduce(max);
    return importance.map((score) => score / maxImportance).toList();
  }

  List<double> predictProbabilities(Tensor input) {
    Tensor output = forward(input);
    return output.data.toList();
  }

  Map<String, dynamic> rocCurve(
      List<Tensor> testData, List<Tensor> testLabels) {
    List<double> trueLabels =
        testLabels.map((tensor) => tensor.data[0]).toList();
    List<double> predictedProbs =
        testData.map((tensor) => predictProbabilities(tensor)[0]).toList();

    List<MapEntry<double, double>> pairs = List.generate(trueLabels.length,
        (index) => MapEntry(predictedProbs[index], trueLabels[index]))
      ..sort((a, b) => b.key.compareTo(a.key));

    int positives = trueLabels.where((label) => label == 1).length;
    int negatives = trueLabels.length - positives;

    List<double> tpr = [0.0];
    List<double> fpr = [0.0];
    int tp = 0;
    int fp = 0;

    for (var pair in pairs) {
      if (pair.value == 1) {
        tp++;
      } else {
        fp++;
      }
      tpr.add(tp / positives);
      fpr.add(fp / negatives);
    }

    return {'tpr': tpr, 'fpr': fpr};
  }

  double auc(Map<String, dynamic> rocCurve) {
    List<double> tpr = rocCurve['tpr'];
    List<double> fpr = rocCurve['fpr'];
    double auc = 0.0;

    for (int i = 1; i < tpr.length; i++) {
      auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2;
    }

    return auc;
  }
}
