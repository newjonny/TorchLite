# TorchLite 🔦

An educational toy-project for a Multilayer Perceptron (MLP) built from scratch in modern C++. 

This project implements the foundational mathematics of neural networks without relying on external machine learning libraries. My aim is to learn fundamental C++ concepts by just using standard library implementations.

## Features
* **From-Scratch Implementation**: Core logic for the Network and its Layers built using standard C++.
* **Forward Propagation**: Calculates weighted sums and applies non-linear activation functions (e.g., ReLU).
* **Automatic Differentiation**: (WIP). Inspired by [Andrej Karpathy's micrograd implementation](https://github.com/karpathy/micrograd).

## Backpropagation and Automatic Differentiation (WIP)
In order to train a MLP, we need to tune all parameters within the neutwork: all weights and biases. In supervised learning, we do this based on the error (or _loss_) between the expected output $y$ and the predicted output $\^{y}$. As  This is done using a mechanism called backpropagation.

Performing backpropagation within a MLP is challenging because we need to calculate derivates along the entire chain of layers in the network.

### A quick recap

**What we know**:
* the error or _loss_ between the last expected output $y$ and the predicted output $\^{y}$ in the last layer
* the current values of weights and biases
* the (non-)linear activation functions used on raw values (see below)

**What we don't know (yet):**
* the error or _loss_ between consecutive layers, apart from the second-to-last and last layer (see above)


### How Automatic Differentiation helps and how it's done

**Idea:** Create a graph of all calculations that we are performing through a neural network. This means that - if we consider **one neuron**, it's raw value $z$ and it's activation $a$ - we take all the calculations in the form of $$ x_1 * w_1 + ... + x_n * w_n$$ and create a calculation graph like this:


![A simple computational graph of one neuron, two inputs and two weights](mermaid-diagram.png)

By traversing this graph from the end, i.e. $a$, to the front, we can backpropagate our final error (that we know) to tune the individual weights (and biases) to train our MLP.

### In more detail: Starting at the end
The first derivative $da/da$ is easy to get as it's by definition just $1$. Traversing back from there, we calculate $da/dz$. For this, we take advantage of the already-derived deriatives of our chosen activation function, whis is a $tanh$, which results in:
$\frac{\partial a}{\partial z} = 1 - \tanh^2(z)$ (cf. [Wikipedia](https://en.wikipedia.org/wiki/Hyperbolic_functions#Derivatives)).

From there, it get's easier again for the next two gradients that we need: $\frac{\partial z}{\partial z1}$ and $\frac{\partial z}{\partial z2}$. As $z = z1 + z2$ is just a simple addition, we'll end up with both local derivatives being $1$.



Forward pro

```mermaid
graph LR
    x1["<b>x1</b><br/><font color='#2E7D32'>val 2</font><br/><font color='#C62828'>grad 0.42</font>"]:::leaf
    w1["<b>w1</b><br/><font color='#2E7D32'>val 1</font><br/><font color='#C62828'>grad 0.84</font>"]:::leaf
    x2["<b>x2</b><br/><font color='#2E7D32'>val 1</font><br/><font color='#C62828'>grad -0.42</font>"]:::leaf
    w2["<b>w2</b><br/><font color='#2E7D32'>val -1</font><br/><font color='#C62828'>grad 0.42</font>"]:::leaf

    mul1(("<b>×</b>")):::mulOp
    mul2(("<b>×</b>")):::mulOp
    add1(("<b>+</b>")):::addOp
    tanh1(("<b>tanh</b>")):::actOp

    z1["<b>z1</b><br/><font color='#2E7D32'>val 2</font><br/><font color='#C62828'>grad 0.42</font>"]:::value
    z2["<b>z2</b><br/><font color='#2E7D32'>val -1</font><br/><font color='#C62828'>grad 0.42</font>"]:::value
    z["<b>z</b><br/><font color='#2E7D32'>val 1</font><br/><font color='#C62828'>grad 0.42</font>"]:::value
    a["<b>a</b><br/><font color='#2E7D32'>val 0.76</font><br/><font color='#C62828'>grad 1</font>"]:::root

    x1 -- "∂z1/∂x1 = w1 = 1"  --> mul1
    w1 -- "∂z1/∂w1 = x1 = 2"  --> mul1
    mul1 --> z1
    x2 -- "∂z2/∂x2 = w2 = -1" --> mul2
    w2 -- "∂z2/∂w2 = x2 = 1"  --> mul2
    mul2 --> z2
    z1 -- "∂z/∂z1 = 1" --> add1
    z2 -- "∂z/∂z2 = 1" --> add1
    add1 --> z
    z -- "∂a/∂z = 1 − tanh²(z) ≈ 0.42" --> tanh1
    tanh1 --> a

    classDef leaf  fill:#FFF59D,stroke:#000,color:#000
    classDef value fill:#FFFFFF,stroke:#000,color:#000
    classDef root  fill:#FFFFFF,stroke:#000,color:#000,stroke-width:3px
    classDef mulOp fill:#FF6B9D,stroke:#000,color:#fff,stroke-width:2px
    classDef addOp fill:#7DF9FF,stroke:#000,color:#000,stroke-width:2px
    classDef actOp fill:#FFD166,stroke:#000,color:#000,stroke-width:2px

```

## Getting Started

### Compilation
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Sources and Further Reading
- [Andrej Karpathy's micrograd implementation](https://github.com/karpathy/micrograd)
- [... and his explanation on YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Michael Kipp's explanation of Automatic Differentiation (DE)](https://michaelkipp.de/deeplearning/AutomatischeDifferenzierung.html)