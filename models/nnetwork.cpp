#include "nnetwork.h"
#include <iostream>
#include <cmath>
#include <random>

Matrix::Matrix(int rows, int cols) : data(rows, std::vector<double>(cols)) {}

Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    Matrix result(a.data.size(), b.data[0].size());
    for (int i = 0; i < a.data.size(); i++) {
        for (int j = 0; j < b.data[0].size(); j++) {
            for (int k = 0; k < a.data[0].size(); k++) {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return result;
}

void Matrix::addBias(const std::vector<double>& bias) {
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].size(); j++) {
            data[i][j] += bias[j];
        }
    }
}

void Matrix::relu() {
    for (auto& row : data) {
        for (auto& x : row) {
            x = std::max(0.0, x);
        }
    }
}

void Matrix::sigmoid() {
    for (auto& row : data) {
        for (auto& x : row) {
            x = 1.0 / (1.0 + exp(-x));
        }
    }
}

NeuralNetwork::NeuralNetwork() :
    weights1(784, 128), weights2(128, 64), weights3(64, 10),
    biases1(128, 0.1), biases2(64, 0.1), biases3(10, 0.1) {}

Matrix NeuralNetwork::forward(const Matrix& input) {
    Matrix hidden1 = Matrix::multiply(input, weights1);
    hidden1.addBias(biases1);
    hidden1.relu();

    Matrix hidden2 = Matrix::multiply(hidden1, weights2);
    hidden2.addBias(biases2);
    hidden2.relu();

    Matrix output = Matrix::multiply(hidden2, weights3);
    output.addBias(biases3);
    output.sigmoid();

    return output;
}
