#include <iostream>
#include "layer.h"
#include "nn.h"
#include "value.h"

void test_nn()
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

    ml::NN nn(4, {3, 2}); // input size 4, first layer with 3 neurons, second layer with 2 neurons
}

void test_value()
{

    std::cout << std::endl;
    ml::Value x(3.0, {}, ml::Op::Add, "x");
    std::cout << x << std::endl;

    // --- Value tests ---
    std::cout << "\n--- Value tests ---\n";

    ml::Value a(2.0, {}, ml::Op::None, "a");
    ml::Value b(5.0, {}, ml::Op::None, "b");
    std::cout << "a: " << a << "\n";
    std::cout << "b: " << b << "\n";

    // binary operators build new graph nodes — operands a and b are NOT mutated
    ml::Value sum  = a + b;
    ml::Value diff = a - b;
    ml::Value prod = a * b;
    ml::Value quot = a / b;

    std::cout << "a + b: " << sum  << "  (expect data=7)\n";
    std::cout << "a - b: " << diff << "  (expect data=-3)\n";
    std::cout << "a * b: " << prod << "  (expect data=10)\n";
    std::cout << "a / b: " << quot << "  (expect data=0.4)\n";

    // chaining builds a deeper graph: (a + b) * a -> Mul node whose left child is the Add node
    ml::Value chained = (a + b) * a;
    std::cout << "(a + b) * a: " << chained << "  (expect data=14)\n";

    // grad is just a public double — set it manually, confirm it prints
    a.grad = 0.5;
    std::cout << "a with grad set: " << a << "\n";

    // confirm operands were never mutated
    std::cout << "a unchanged: " << a << "  (expect data=2)\n";
    std::cout << "b unchanged: " << b << "  (expect data=5)\n";
}

int main()
{
    test_value();
    return 0;
}