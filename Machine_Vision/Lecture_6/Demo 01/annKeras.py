import numpy as np
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_data(filepath):
    """Load and preprocess the diabetes dataset."""
    data = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            data.append([float(val) for val in row])
    
    data = np.array(data)
    X = data[:, :-1]  # Features
    y = data[:, -1:]  # Labels
    
    return X, y


def normalize_data(X_train, X_test):
    """Normalize features using mean and standard deviation from training set."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1
    
    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std
    
    return X_train_normalized, X_test_normalized, mean, std


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """Split data into training, validation, and test sets."""
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    train_end = int(train_ratio * n_samples)
    val_end = int((train_ratio + val_ratio) * n_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_model(input_dim, layer_sizes=[16, 8], learning_rate=0.01):
    """
    Create a Keras Sequential model for binary classification.
    
    Args:
        input_dim: Number of input features
        layer_sizes: List of neurons in each hidden layer
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='Diabetes_Prediction_ANN')
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,), name='input_layer'))
    
    # Hidden layers
    for i, units in enumerate(layer_sizes):
        model.add(layers.Dense(
            units=units,
            activation='relu',
            kernel_initializer='he_normal',
            name=f'hidden_layer_{i+1}'
        ))
    
    # Output layer
    model.add(layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer='glorot_uniform',
        name='output_layer'
    ))
    
    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set."""
    # Get predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    
    # Confusion matrix
    true_positives = np.sum((y_pred == 1) & (y_test == 1))
    true_negatives = np.sum((y_pred == 0) & (y_test == 0))
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': {
            'TP': int(true_positives),
            'TN': int(true_negatives),
            'FP': int(false_positives),
            'FN': int(false_negatives)
        }
    }


def main():
    """Main function to train and evaluate the neural network."""
    print("=" * 60)
    print("Diabetes Prediction using TensorFlow Keras")
    print("Artificial Neural Network Implementation")
    print("=" * 60)
    print()
    
    # Load data
    print("Loading dataset...")
    X, y = load_data('diabetes.csv')
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Positive cases: {int(np.sum(y))} ({np.mean(y) * 100:.1f}%)")
    print()
    
    # Split data
    print("Splitting data into train/validation/test sets...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print()
    
    # Normalize data
    print("Normalizing features...")
    X_train, X_test, mean, std = normalize_data(X_train, X_test)
    X_val = (X_val - mean) / std
    print()
    
    # Create model
    print("Creating neural network...")
    input_dim = X_train.shape[1]
    layer_sizes = [16, 8]  # Hidden layers
    model = create_model(input_dim=input_dim, layer_sizes=layer_sizes, learning_rate=0.01)
    
    # Display model architecture
    print("\nModel Architecture:")
    print("-" * 60)
    model.summary()
    print("-" * 60)
    print()
    
    # Count neurons
    total_neurons = input_dim
    for units in layer_sizes:
        total_neurons += units
    total_neurons += 1  # Output layer
    
    print(f"Network Details:")
    print(f"  - Input neurons: {input_dim}")
    print(f"  - Hidden layers: {len(layer_sizes)}")
    for i, units in enumerate(layer_sizes):
        print(f"    - Hidden layer {i+1}: {units} neurons")
    print(f"  - Output neurons: 1")
    print(f"  - Total neurons: {total_neurons}")
    print()
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=0.0001,
        verbose=1
    )
    
    # Train the model
    print("Training the model...")
    print("-" * 60)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    print("-" * 60)
    print()
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        X_test, y_test, verbose=0
    )
    
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\nTest Set Results:")
    print("=" * 60)
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print()
    print("Confusion Matrix:")
    print(f"  True Positives:  {metrics['confusion_matrix']['TP']}")
    print(f"  True Negatives:  {metrics['confusion_matrix']['TN']}")
    print(f"  False Positives: {metrics['confusion_matrix']['FP']}")
    print(f"  False Negatives: {metrics['confusion_matrix']['FN']}")
    print("=" * 60)
    
    # Make sample predictions
    print("\nSample Predictions (first 10 test samples):")
    print("-" * 60)
    predictions_prob = model.predict(X_test[:10], verbose=0)
    predictions = (predictions_prob >= 0.5).astype(int)
    
    for i in range(10):
        print(f"Sample {i+1}: Predicted: {predictions[i][0]}, "
              f"Probability: {predictions_prob[i][0]:.4f}, Actual: {int(y_test[i][0])}")
    print("-" * 60)
    
    # Training history summary
    print("\nTraining History:")
    print("-" * 60)
    print(f"Total epochs trained: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print("-" * 60)
    
    # Save the model
    #print("\nSaving model...")
    #model.save('diabetes_model.keras')
    #print("Model saved to: diabetes_model.keras")
    
    return model, history, metrics


if __name__ == "__main__":
    model, history, metrics = main()