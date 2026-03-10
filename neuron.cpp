/**#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <algorithm>
#include "neuron.h"

using namespace std;

namespace ml
{
    Neuron::Neuron(vector<double> weights, double bias)
    {
        cout << "\nCreated Neuron";
        weights_ = weights;
        bias_ = bias;
    }

    Neuron::~Neuron()
    {
        cout << "\nDeconstructed Neuron";
    }

    double Neuron::activate(vector<double> &inputs, string &act_function)
    {
        const double weigh_sum = std::inner_product(inputs.begin(), inputs.end(), weights_.begin(), 0.0);
        const double z = weigh_sum + bias_; // raw input

        double act;
        if (act_function == "relu")
        {
            act = max(z, 0.0);
        }
        // TODO add more activation function types
        return act;
    }
}*/