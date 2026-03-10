#include <iostream>
#include "layer.h"

int main()
{
    ml::Layer layer(3, 4); // 3 neurons, each with 4 inputs
    std::vector<double> inputs = {1.0, 2.0, 3.0, 4.0};
    std::string activation_function = "relu";
    std::vector<double> outputs = layer.forward(inputs, activation_function);
    std::cout << "\nLayer outputs: ";
    for (const auto& output : outputs)
    {
        std::cout << output << " ";
    }
}