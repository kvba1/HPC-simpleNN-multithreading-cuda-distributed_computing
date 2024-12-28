#include <vector>
#include <pthread.h>
#include <algorithm>
#include <cmath>
#include "nnetwork_parallel.h"
#include <thread>
#include <iostream>

Matrix::Matrix(int rows, int cols) : data(rows, std::vector<double>(cols)) {}

const int MAX_THREADS = std::max(1u, std::thread::hardware_concurrency());

void* multiplyRowRange(void* arg);

struct ThreadData {
    const Matrix* a;
    const Matrix* b;
    Matrix* result;
    int start_row;
    int end_row;
};

Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    int num_threads = std::min({MAX_THREADS, static_cast<int>(a.data.size()), 8});
    Matrix result(a.data.size(), b.data[0].size());
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int rows_per_thread = (a.data.size() + num_threads - 1) / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].a = &a;
        thread_data[i].b = &b;
        thread_data[i].result = &result;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = std::min((i + 1) * rows_per_thread, static_cast<int>(a.data.size()));
        pthread_create(&threads[i], NULL, multiplyRowRange, (void*)&thread_data[i]);
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    return result;
}

void* multiplyRowRange(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    const Matrix& a = *data->a;
    const Matrix& b = *data->b;
    Matrix& result = *data->result;

    for (int i = data->start_row; i < data->end_row; ++i) {
        for (int j = 0; j < b.data[0].size(); ++j) {
            double sum = 0;
            for (int k = 0; k < a.data[0].size(); ++k) {
                sum += a.data[i][k] * b.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    pthread_exit(NULL);
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