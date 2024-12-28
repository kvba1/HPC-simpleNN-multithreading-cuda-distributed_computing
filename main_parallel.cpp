#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "models/nnetwork_parallel.h"

int main() {

    NeuralNetwork neuralNetwork;

    Matrix input(64, 784);

    for (auto& row : input.data) {
        std::fill(row.begin(), row.end(), 1.0);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix output = neuralNetwork.forward(input);
    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << "Output of the neural network:" << std::endl;
    for (const auto& row : output.data) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;
    return 0;
}
