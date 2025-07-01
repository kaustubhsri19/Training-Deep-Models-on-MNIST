# Training-Deep-Models-on-MNIST

## Project 1 – Part 2: MNIST Digit Classifier with TensorFlow/Keras

---

### 🎯 Goal

Rebuild the handwritten digit classification model using **Keras** to:
- Leverage high-level APIs for faster prototyping.
- Improve accuracy and efficiency with better initialization, optimizers, and training tools.

---

### ⚙️ Tools Used

- **TensorFlow/Keras**
- `Sequential` model
- `Dense` layers
- Optimizer: `Adam`
- Loss: `CategoricalCrossentropy`
- Metrics: `Accuracy`

---

### 📦 Dataset

- **MNIST dataset** loaded using `fetch_openml()` from `sklearn.datasets`.
- Images flattened from `(28, 28)` to `784` features.
- Pixel values normalized to `[0, 1]`.
- Labels converted to **one-hot encoded** vectors of shape `(n_samples, 10)`.

---

### 🧠 Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
```

- **Dense(64, relu):** Hidden layer with 64 neurons
- **Dense(32, relu):** Hidden layer with 32 neurons
- **Dense(10, softmax):** Output layer for 10 digit classes

---

### 🔁 Training Process

- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Batch size: 32 (default)
- Epochs: 100
- Validation split: 20%

Training command:
```python
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
```

---

### 📊 Evaluation & Prediction

- Validation accuracy: ~81%
- Used `.predict()` for class probabilities
- Final predictions with `np.argmax`
- Submission CSV created for Kaggle

---

### 📈 Result

- Achieved **~81% accuracy** with a simple dense neural network.
- Potential improvements:
  - **Dropout** (to prevent overfitting)
  - **Batch normalization**
  - **More complex architectures** (e.g., CNNs)

---

### 🧠 Concepts Practiced

- Model building with Keras Sequential API
- Activation functions (`ReLU`, `Softmax`)
- Loss functions and optimizers
- Validation split and training loop
- Prediction and submission formatting

---

### ✅ Summary Table

| Feature           | NumPy (Part 1)   | Keras (Part 2)          |
| ----------------- | ---------------- | ----------------------- |
| Code Control      | Full manual      | High-level API          |
| Training Speed    | Slower           | Much faster             |
| Accuracy Achieved | ~90.25%          | ~81% (can be improved)  |
| Learning Focus    | Internals (math) | Framework usage         |