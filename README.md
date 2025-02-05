# Quantum-Inspired Echo State Network for Advanced Time-Series Forecasting

## 📌 Project Overview

This repository presents an advanced **Quantum-Inspired Echo State Network (ESN)** for **time-series forecasting**, specifically applied to stock price prediction. The model leverages **reservoir computing** along with a **quantum-inspired feature mapping** to enhance predictive accuracy and performance. The primary dataset used is historical stock data, but the model is designed to be adaptable for other complex time-series data.

## 🔹 What is an Echo State Network (ESN)?

An **Echo State Network (ESN)** is a type of **Recurrent Neural Network (RNN)** that utilizes a **reservoir computing** approach. Unlike traditional RNNs, ESNs:
- Do **not require backpropagation through time (BPTT)** for training.
- Maintain a large, **fixed reservoir** of randomly initialized recurrent neurons.
- Only train the **output layer** using a simple regression model (e.g., Ridge regression).

### 🔹 How ESNs Work:
1. **Input-to-Reservoir Mapping**: The input signal is projected into a **high-dimensional reservoir** with random, sparse connections.
2. **Reservoir Dynamics**: The reservoir maintains a memory of past inputs through **nonlinear transformations**.
3. **Readout Layer Training**: Only the weights of the output layer are optimized, typically using **ridge regression**.

### 🔢 **Mathematical Formulation of ESN**
Let:
- **\( x_t \)** be the input at time **\( t \)**.
- **\( h_t \)** be the reservoir state at time **\( t \)**.
- **\( W_{in} \)** and **\( W \)** be the input-to-reservoir and reservoir-to-reservoir weight matrices, respectively.
- **\( y_t \)** be the predicted output at time **\( t \)**.

The reservoir state is updated as:

\[
h_t = (1 - \alpha) h_{t-1} + \alpha \tanh(W_{in} x_t + W h_{t-1})
\]

where **\( \alpha \)** is the **leak rate** controlling how much of the past state is retained.

The output is computed as:

\[
y_t = W_{out} h_t
\]

where **\( W_{out} \)** is learned using **ridge regression**:

\[
W_{out} = (H^T H + \lambda I)^{-1} H^T Y
\]

where **\( H \)** is the collected reservoir states, and **\( \lambda \)** is a regularization parameter.

## 🔹 What is Quantum-Inspired Feature Mapping?

Quantum computing introduces unique transformations that can be applied to classical AI models. In this project, we use a **quantum-inspired feature mapping** to enhance ESN performance.

### 🔢 **Mathematical Formulation**
Instead of using raw reservoir states, we apply a **nonlinear quantum-inspired transformation**:

\[
\phi(h_t) = \sin(h_t)
\]

This transformation is inspired by **quantum wave functions**, where trigonometric functions model probability amplitudes in quantum systems.

### ✨ **Why Use Quantum-Inspired Mapping?**
- Captures **nonlinear relationships** more effectively.
- Helps in maintaining a richer **feature representation**.
- Inspired by **quantum machine learning**, it enhances generalization in time-series forecasting.

## 🔹 Project Implementation

### 📂 **Dataset**
- The model is trained on **Amazon (AMZN) stock price data**.
- The dataset contains daily **adjusted close prices**.

### 🛠 **Preprocessing Steps**
1. **Load and clean the dataset**.
2. **Normalize the data** using **Min-Max scaling** (range: -1 to 1).
3. **Create time-series sequences** for training/testing.
4. **Split the data** into training (80%) and testing (20%).

### 🏗 **Model Architecture**
- **Reservoir Size**: 300 neurons
- **Spectral Radius**: 0.95
- **Sparsity**: 10%
- **Leak Rate**: 0.6
- **Readout Layer**: Ridge Regression (L2 Regularization: 0.5)

### 📊 **Performance Metrics**
- **Mean Squared Error (MSE)** to evaluate prediction accuracy.

## 🔹 Results & Visualization
- The model successfully **predicts stock prices** based on past values.
- A visualization of **actual vs. predicted prices** is generated using **Matplotlib**.
  
# Project Title

🔗 **Authors:** Ali Ibraheem  
📧 **Contact:** aliibraheem516@gmail.com  
🧠 **Keywords:** AI, Machine Learning, Quantum Computing, Time-Series Forecasting, Echo State Networks (ESN)
