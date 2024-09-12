// import 'dart:math';
// import 'package:dart_ml/core/matrices.dart';
// import 'package:dart_ml/core/tensors.dart';
// import 'package:dart_ml/core/optimizers.dart';
// import 'package:dart_ml/model_training/sequential_model.dart';
// import 'package:dart_ml/visualization/visualization.dart';
// import 'package:dart_ml/visualization/visualizations.dart';

// void main() {
//   // 1. Create sample data
//   // 1. Create sample data
//   final inputs = Matrix([
//     [1.0, 2.0],
//     [2.0, 4.0],
//     [3.0, 6.0],
//     [4.0, 8.0]
//   ]);
//   final targets = Matrix([
//     [3.0],
//     [5.0],
//     [7.0],
//     [9.0]
//   ]);

//   // 2. Define the model with layers
//   final layers = [
//     LinearLayer(2, 5), // Input size 2, output size 5
//     LinearLayer(5, 1)  // Input size 5, output size 1
//   ];
//   final model = SequentialModel(layers);

//   // 3. Compile the model with optimizer and loss function
//   final optimizer = GradientDescentOptimizer(learningRate: 0.01);

//   // 4. Train the model
//   final epochs = 1000;
//   for (var epoch = 0; epoch < epochs; epoch++) {
//     // Forward pass
//     final predictions = model.forward(inputs);

//     // Compute loss (mean squared error)
//     final loss = (predictions - targets).elementwiseOperation((x) => x * x).mean();
//     print('Epoch $epoch: Loss = $loss');

//     // Compute gradients
//     final lossGradient = (predictions - targets).elementwiseOperation((x) => 2 * x);
//     model.backward(lossGradient);

//     // Update parameters
//     model.updateParameters(optimizer.learningRate);
//   }

//   // 5. Make predictions
//   final predictions = model.forward(inputs);

//   // 6. Visualize results
//   final visualization = Visualization();
//   final plotData = [
//     {'x': inputs.getColumn(0).toList(), 'y': targets.getColumn(0).toList(), 'label': 'True Values'},
//     {'x': inputs.getColumn(0).toList(), 'y': predictions.toList(), 'label': 'Predictions'}
//   ];
//   visualization.plotLinePlot(plotData, 'Model Predictions vs True Values');
// }
