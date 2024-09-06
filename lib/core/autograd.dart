import 'dart:typed_data';

import 'tensors.dart';

class Variable {
  Tensor data;
  Tensor? grad;
  Function? _backward;
  List<Variable> _prevNodes = [];

  Variable(this.data);

  void backward() {
    if (grad == null) {
      grad = Tensor.ones(data.shape);
    }

    List<Variable> topoOrder = [];
    Set<Variable> visited = {};

    void buildTopoOrder(Variable v) {
      if (!visited.contains(v)) {
        visited.add(v);
        for (var prevNode in v._prevNodes) {
          buildTopoOrder(prevNode);
        }
        topoOrder.add(v);
      }
    }

    buildTopoOrder(this);

    for (var v in topoOrder.reversed) {
      if (v._backward != null) {
        v._backward!();
      }
    }
  }
}

Variable add(Variable a, Variable b) {
  var out = Variable(a.data + b.data);
  out._prevNodes = [a, b];

  out._backward = () {
    if (out.grad != null) {
      if (a.grad == null) a.grad = Tensor.zeros(a.data.shape);
      if (b.grad == null) b.grad = Tensor.zeros(b.data.shape);
      a.grad = a.grad! + out.grad!;
      b.grad = b.grad! + out.grad!;
    }
  };

  return out;
}

Variable multiply(Variable a, Variable b) {
  var out = Variable(a.data * b.data);
  out._prevNodes = [a, b];

  out._backward = () {
    if (out.grad != null) {
      if (a.grad == null) a.grad = Tensor.zeros(a.data.shape);
      if (b.grad == null) b.grad = Tensor.zeros(b.data.shape);
      a.grad = a.grad! + (out.grad! * b.data);
      b.grad = b.grad! + (out.grad! * a.data);
    }
  };

  return out;
}


