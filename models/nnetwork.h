#include <vector>

// Matrix class declaration
class Matrix {
public:
    std::vector<std::vector<double>> data;

    Matrix(int rows, int cols);

    static Matrix multiply(const Matrix& a, const Matrix& b);
    void addBias(const std::vector<double>& bias);
    void relu();
    void sigmoid();
};

class NeuralNetwork {
private:
    Matrix weights1, weights2, weights3;
    std::vector<double> biases1, biases2, biases3;

public:
    NeuralNetwork();
    Matrix forward(const Matrix& input);
};
