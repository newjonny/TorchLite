/**#include <vector>
#include <string>

namespace ml
{
    class Neuron
    {
        private:
            std::vector<double> weights_;
            double bias_;
        public:
            Neuron(/* args */);
            ~Neuron();

            double activate(std::vector<double>& inputs_, string& act_function); // pass inputs by ref

        
        Neuron::Neuron(std::vector<double> weights, double bias)
        {
        }
        
        Neuron::~Neuron()
        {
        }
    };
}*/