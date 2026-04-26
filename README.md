# TorchLite

An educational toy-project for a Multilayer Perceptron (MLP) built from scratch in modern C++. 

This project implements the foundational mathematics of neural networks without relying on external machine learning libraries. My aim is to learn fundamental C++ concepts by just using standard library implementations.

## Features
* **From-Scratch Implementation**: Core logic for the Network and its Layers built using standard C++.
* **Forward Propagation**: Calculates weighted sums and applies non-linear activation functions (e.g., ReLU).
* **Automatic Differentiation**: (WIP). Inspired by [Andrej Karpathy's micrograd implementation](https://github.com/karpathy/micrograd).

## Getting Started

### Compilation
```bash
mkdir build
cd build
cmake ..
cmake --build .```