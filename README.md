# TorchLite

An educational toy-project for a Multilayer Perceptron (MLP) built from scratch in modern C++. 

This project implements the foundational mathematics of neural networks without relying on external machine learning libraries. My aim is to learn fundamental C++ concepts by just using standard library implementations.

## Features
* **From-Scratch Implementation**: Core logic for the Network and its Layers built using standard C++.
* **Forward Propagation**: Calculates weighted sums and applies non-linear activation functions (e.g., ReLU).
* **Modern C++ Practices**: Utilizes `<random>` for mathematically sound weight initialization and standard containers for data management.

## Getting Started

### Prerequisites
* A C++11 (or higher) compatible compiler (GCC, Clang, MSVC).
* CMake (optional, but recommended for building).

### Compilation
```bash
mkdir build
cd build
cmake ..
cmake --build .```