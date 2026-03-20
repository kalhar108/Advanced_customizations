# Deep Learning: Data Augmentation, Regularization & Advanced Constructs

**Author:** Kalhar Mayurbhai Patel  
**Student ID:** 019140511

---

## Overview

This repository contains a comprehensive collection of Jupyter notebooks demonstrating deep learning techniques across two major parts:

- **Part 1** — Regularization, generalization, and data augmentation techniques (11 focused notebooks, a–k)
- **Part 2** — Advanced Keras and PyTorch constructs (1 unified notebook covering 12 topics)

Every notebook includes implementations in **both TensorFlow/Keras and PyTorch**, with A/B comparisons, visualizations, and explanations.

---

## Repository Structure

```
.
├── README.md
├── Part1_Notebooks/
│   ├── 1a_L1_L2_Regularization.ipynb
│   ├── 1b_Dropout.ipynb
│   ├── 1c_EarlyStopping.ipynb
│   ├── 1d_MonteCarlo_Dropout.ipynb
│   ├── 1e_Weight_Initialization.ipynb
│   ├── 1f_BatchNorm.ipynb
│   ├── 1g_Custom_Dropout_Regularization.ipynb
│   ├── 1h_Callbacks_TensorBoard.ipynb
│   ├── 1i_KerasTuner.ipynb
│   ├── 1j_KerasCV_Augmentation.ipynb
│   └── 1k_MultiDomain_Augmentation.ipynb
└── Part2_Notebook/
    └── Part2_Advanced_DL_Constructs.ipynb
```

---

## Part 1: Regularization & Data Augmentation (Notebooks a–k)

Each notebook is self-contained, runs on Google Colab, and includes TF + PyTorch implementations.

### 1a — L1 & L2 Regularization
Compares baseline (no regularization), L1, L2, and ElasticNet on a synthetic classification task. Shows how L1 promotes sparsity while L2 keeps weights small. Includes side-by-side validation curves.

### 1b — Dropout
A/B test across dropout rates (0.0, 0.2, 0.5, 0.7). Demonstrates the sweet spot for reducing overfitting without underfitting. Visualizes training vs validation loss divergence.

### 1c — Early Stopping
Implements Keras `EarlyStopping` callback and a manual early stopping class for PyTorch. Compares training with and without early stopping over 200 epochs, showing the compute savings and overfitting prevention.

### 1d — Monte Carlo Dropout
Builds a custom `MCDropout` layer that stays active during inference. Runs 100 forward passes to generate prediction distributions, then visualizes mean predictions and uncertainty maps on a moons dataset.

### 1e — Weight Initialization
Tests 5 initialization strategies (He, Glorot, LeCun, Zeros) paired with matching activations (ReLU, Tanh, SELU). Includes a reference table for when to use each initializer.

### 1f — Batch Normalization
Compares deep networks with and without BatchNorm. Demonstrates BN placement (before activation), shows faster convergence and mild regularization effect.

### 1g — Custom Dropout & Custom Regularization
Implements from scratch:
- **Alpha Dropout** — maintains self-normalizing property for SELU networks
- **Spatial/Channel Dropout** — drops entire feature channels
- **Target Weight Regularizer** — penalizes weights far from a target value

### 1h — Callbacks & TensorBoard
Demonstrates 7 Keras callbacks (EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, LRScheduler, Custom). PyTorch section uses `SummaryWriter` for scalar, histogram, and image logging.

### 1i — Keras Tuner
Defines a hypermodel with tunable layers, units, dropout, batch norm, and learning rate. Runs both `RandomSearch` and `Hyperband` tuning strategies with comparison.

### 1j — KerasCV Data Augmentation
Uses `keras_cv` layers (RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast) integrated into a `tf.data` pipeline. A/B tests augmented vs non-augmented training on CIFAR-10.

### 1k — Multi-Domain Data Augmentation
Covers augmentation across 5 domains:

| Domain | Techniques |
|--------|-----------|
| **Image** | Rotation, zoom, translation, perspective (TF + PyTorch) |
| **Text** | Synonym replacement, character insertion via `nlpaug` |
| **Time Series** | Jittering, scaling, time warping, window slicing, magnitude warping |
| **Tabular** | Gaussian noise injection, SMOTE for class imbalance |
| **Audio** | Noise addition, time stretching, pitch shifting |

---

## Part 2: Advanced Deep Learning Constructs (Single Notebook)

One comprehensive notebook covering 12 advanced topics, each implemented in both TensorFlow and PyTorch using Fashion MNIST.

| # | Topic | What's Implemented |
|---|-------|-------------------|
| 1 | **Custom LR Scheduler** | OneCycle policy — linear warmup + cosine anneal |
| 2 | **Custom Dropout** | MC Alpha Dropout for SELU networks with uncertainty estimation |
| 3 | **Custom Normalization** | MaxNormDense — constrains weight norms after each update |
| 4 | **TensorBoard** | Scalar logging, weight histograms, model graphs, sample images |
| 5 | **Custom Loss** | Huber Loss — robust to outliers (quadratic small, linear large) |
| 6 | **Custom Components** | Custom activation (LeakyReLU), initializer (Glorot), L1 regularizer, positive weight constraint |
| 7 | **Custom Metric** | Streaming Huber metric with accumulation across batches |
| 8 | **Custom Layers** | GaussianNoise, MyDense (from scratch), ExponentialLayer |
| 9 | **Custom Model** | Residual network with skip connections (ResidualBlock + ResidualRegressor) |
| 10 | **Custom Optimizer** | Momentum SGD built from scratch without using built-in optimizers |
| 11 | **Custom Training Loop** | Manual gradient computation with `GradientTape` (TF) and `loss.backward()` (PT) |
| 12 | **W&B Integration** | Experiment tracking with Weights & Biases — config logging, metric tracking, model watching |

---

## How to Run

1. **Google Colab (recommended):** Upload any notebook to [colab.research.google.com](https://colab.research.google.com) and run all cells. GPU runtime recommended for faster execution.

2. **Local:** Install dependencies:
   ```bash
   pip install tensorflow torch torchvision scikit-learn matplotlib keras-tuner keras-cv nlpaug wandb
   ```
   Then open notebooks with `jupyter notebook`.

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `tensorflow` | Keras models, layers, callbacks |
| `torch` / `torchvision` | PyTorch models, augmentation transforms |
| `scikit-learn` | Data generation, preprocessing, train/test splits |
| `keras-tuner` | Automated hyperparameter search |
| `keras-cv` | GPU-accelerated image augmentation |
| `nlpaug` | Text data augmentation |
| `wandb` | Experiment tracking and visualization |
| `matplotlib` | Plotting and visualization |

---

## Video Walkthrough

A detailed video walkthrough is provided covering:
- Code explanation for each notebook
- Live execution and output demonstration
- Discussion of key concepts and when to use each technique

---

## References

- Géron, A. — *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.)
- [tensorflow.org documentation](https://www.tensorflow.org)
- [PyTorch documentation](https://pytorch.org/docs)
- [KerasCV](https://keras.io/keras_cv/)
- [nlpaug](https://github.com/makcedward/nlpaug)
- [Data Augmentation Review](https://github.com/AgaMiko/data-augmentation-review)
