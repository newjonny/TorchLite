#pragma once
#include <vector>
#include <string>

namespace ml
{
    class Layer
    {
        private:
            int num_neurons_;
            std::vector<std::vector<double>> weights_;
            std::vector<double> biases_;

            double activate(double z, std::string &act_function);
            std::vector<std::vector<double>> initialize_weights(int &num_neurons, int &num_inputs_per_neuron);
        public:
            Layer(int num_neurons, int num_inputs_per_neuron);
            ~Layer();

            std::vector<double> forward(std::vector<double>& inputs, std::string& act_function);
            std::vector<double> backward(std::vector<double>& gradients); // backprop is a task for Jonas from the future
    };
}