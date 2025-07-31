"""
train_model_colab.py

This script trains a simple CNN image classifier using CIFAR-10 dataset
and saves the trained model as 'sample_model.h5'.

Run this file in Google Colab.
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Save the model to .h5
model.save("sample_model.h5")

print("✅ Training complete. Model saved as 'sample_model.h5'.")

# Optional: Add code to download the model in Google Colab
try:
    from google.colab import files
    files.download("sample_model.h5")
except:
    print("If not in Colab, please manually download 'sample_model.h5'")
