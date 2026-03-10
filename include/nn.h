#include <iostream>
#include <vector>
#include "layer.h"


namespace ml
{
    class NN // Neural Network
    {
        public:
            NN(int input_size, std::vector<int> layer_sizes);
            ~NN();
        private:
            int input_size_;
            std::vector<Layer> layers_;

    };

    NN::NN(int input_size, std::vector<int> layer_sizes)
    {
        std::cout << "\nCreate Neural Network";

        input_size_ = input_size;
        for(int i = 0; i < static_cast<int>(layer_sizes.size()); i++) // layer_sizes.size() is the number of layers in the network
        {
            int num_neurons = layer_sizes.at(i);
            // for the first layer, the number of inputs per neuron is the input size,
            // for subsequent layers it's the number of neurons in the previous layer
            int num_inputs_per_neuron = (i == 0) ? input_size : layer_sizes.at(i-1);
            
            // create a new Layer and add it to the layers_ vector, added to the end of the vector
            layers_.emplace_back(num_neurons, num_inputs_per_neuron);
        }
    }

    NN::~NN()
    {
        std::cout << "\nDeconstruct Neural Network";
    }
}