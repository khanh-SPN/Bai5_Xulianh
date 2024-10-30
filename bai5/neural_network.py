import tensorflow as tf
from data_preparation import load_iris_data, load_dental_images
from evaluation import evaluate_model

def build_neural_network(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# IRIS dataset
X_train_iris, X_test_iris, y_train_iris, y_test_iris = load_iris_data()
model_iris = build_neural_network((X_train_iris.shape[1],), len(set(y_train_iris)))
model_iris.fit(X_train_iris, y_train_iris, epochs=10, batch_size=16, validation_data=(X_test_iris, y_test_iris))

# Dental images dataset
X_train_dental, X_test_dental, y_train_dental, y_test_dental = load_dental_images()
model_dental = build_neural_network((X_train_dental.shape[1],), len(set(y_train_dental)))
model_dental.fit(X_train_dental, y_train_dental, epochs=10, batch_size=16, validation_data=(X_test_dental, y_test_dental))