#include <iostream>
#include <vector>
#include <string>
#include <random>
#include "layer.h"

namespace ml
{

    /**
     * TODO: Consider storing the weight matrix instead of putting it into different Neuron objects!
     */

    Layer::Layer(int num_neurons, int num_inputs_per_neuron)
    {
        std::cout << "\nCreated Layer";
        num_neurons_ = num_neurons;
        weights_ = initialize_weights(num_neurons, num_inputs_per_neuron);
        biases_ = std::vector<double>(num_inputs_per_neuron, 0.0); // set bias to 0 for now
    }

    Layer::~Layer()
    {
        std::cout << "\nDeconstruct Layer";
    }

    std::vector<double> Layer::forward(std::vector<double> &inputs, std::string &act_function)
    {
        std::cout << "\nForward pass through Layer";
        std::vector<double> outputs;
        outputs.reserve(num_neurons_); // reserve space for as many outputs as neurons in the layer

        // Forward pass:
        for (int i = 0; i < num_neurons_; i++) // no of neurons are the number of rows in the weight matrix
        {
            double z = 0.0;
            for(int j = 0; j < static_cast<int>(inputs.size()); j++) // no of summands reflec the no of input neurons
            {
                std::cout << "\nCalculating z for neuron " << i << ": adding weight " << weights_.at(i).at(j) << " * input " << inputs.at(j);
                z += weights_.at(i).at(j) * inputs.at(j);
            }
            const double z_with_bias = z + biases_.at(i);

            outputs.push_back(this->activate(z_with_bias, act_function)); // activate raw input
        }
        return outputs;
    }

    std::vector<double> Layer::backward(std::vector<double> &gradients)
    {
        // Placeholder for backpropagation.
        return {};
    }

    double Layer::activate(double z, std::string &act_function)
    {
        double act;
        if (act_function == "relu")
        {
            act = std::max(z, 0.0);
        }
        else
        {
            std::cerr << "\nUnsupported activation function: " << act_function;
            act = z; // just return raw input if given act fct not supported
        }
        // TODO add more activation function types
        return act;
    }

    std::vector<std::vector<double>> Layer::initialize_weights(int &num_neurons, int &num_inputs_per_neuron)
    {
        std::vector<std::vector<double>> rand_weights;

        // This is a weird 3-step way in C++ to generate random numbers:
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0); // between -1 and 1

        for (int i = 0; i < num_neurons; i++)
        {
            std::vector<double> rand_weights_row;
            for(int j = 0; j < num_inputs_per_neuron; j++)
            {
                const double random_weight = dist(gen);
                rand_weights_row.push_back(random_weight);
            }
            rand_weights.push_back(rand_weights_row);
        }

        return rand_weights;
    }
}