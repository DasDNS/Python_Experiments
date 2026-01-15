import numpy as np
import csv

# Set random seed for reproducibility
np.random.seed(42)


class NeuralNetwork:
    """
    A basic Artificial Neural Network implemented from scratch using only NumPy.
    Supports multiple hidden layers with customizable architecture.
    """
    
    def __init__(self, layer_sizes):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
                        e.g., [8, 16, 8, 1] means 8 input features, two hidden layers 
                        with 16 and 8 neurons, and 1 output neuron
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases using He initialization
        for i in range(self.num_layers - 1):
            # He initialization for better convergence with ReLU
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip values to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        """Derivative of sigmoid function"""
        return a * (1 - a)
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU function"""
        return (z > 0).astype(float)
    
    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            activations: List of activations for each layer
            z_values: List of pre-activation values for each layer
        """
        activations = [X]
        z_values = []
        
        for i in range(self.num_layers - 1):
            # Compute pre-activation values
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Apply activation function
            if i == self.num_layers - 2:  # Output layer
                a = self.sigmoid(z)
            else:  # Hidden layers
                a = self.relu(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def compute_cost(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            cost: Average loss over all samples
        """
        m = y_true.shape[0]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def backward_propagation(self, X, y, activations, z_values):
        """
        Perform backward propagation to compute gradients.
        
        Args:
            X: Input data
            y: True labels
            activations: List of activations from forward propagation
            z_values: List of pre-activation values from forward propagation
            
        Returns:
            weight_gradients: List of gradients for weights
            bias_gradients: List of gradients for biases
        """
        m = X.shape[0]
        weight_gradients = []
        bias_gradients = []
        
        # Compute output layer error
        delta = activations[-1] - y
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            # Compute gradients
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            # Compute error for previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(z_values[i - 1])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients, learning_rate):
        """
        Update weights and biases using gradient descent.
        
        Args:
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
            learning_rate: Learning rate for gradient descent
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=1000, learning_rate=0.01, 
              batch_size=32, verbose=True):
        """
        Train the neural network using mini-batch gradient descent.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Size of mini-batches
            verbose: Whether to print training progress
        """
        n_samples = X_train.shape[0]
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward propagation
                activations, z_values = self.forward_propagation(X_batch)
                
                # Backward propagation
                weight_gradients, bias_gradients = self.backward_propagation(
                    X_batch, y_batch, activations, z_values
                )
                
                # Update parameters
                self.update_parameters(weight_gradients, bias_gradients, learning_rate)
            
            # Calculate metrics every 50 epochs
            if epoch % 50 == 0 or epoch == epochs - 1:
                # Training metrics
                train_activations, _ = self.forward_propagation(X_train)
                train_predictions = train_activations[-1]
                train_loss = self.compute_cost(y_train, train_predictions)
                train_acc = self.compute_accuracy(y_train, train_predictions)
                
                # Validation metrics
                val_activations, _ = self.forward_propagation(X_val)
                val_predictions = val_activations[-1]
                val_loss = self.compute_cost(y_val, val_predictions)
                val_acc = self.compute_accuracy(y_val, val_predictions)
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                
                if verbose:
                    print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | "
                          f"Val Acc: {val_acc:.4f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X: Input data
            
        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Predicted probabilities
        """
        activations, _ = self.forward_propagation(X)
        probabilities = activations[-1]
        predictions = (probabilities >= 0.5).astype(int)
        return predictions, probabilities
    
    def compute_accuracy(self, y_true, y_pred):
        """
        Compute classification accuracy.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            accuracy: Classification accuracy
        """
        predictions = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(predictions == y_true)
        return accuracy


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


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set."""
    predictions, probabilities = model.predict(X_test)
    
    # Calculate metrics
    accuracy = np.mean(predictions == y_test)
    
    # Confusion matrix
    true_positives = np.sum((predictions == 1) & (y_test == 1))
    true_negatives = np.sum((predictions == 0) & (y_test == 0))
    false_positives = np.sum((predictions == 1) & (y_test == 0))
    false_negatives = np.sum((predictions == 0) & (y_test == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': {
            'TP': true_positives,
            'TN': true_negatives,
            'FP': false_positives,
            'FN': false_negatives
        }
    }


def main():
    """Main function to train and evaluate the neural network."""
    print("=" * 60)
    print("Diabetes Prediction using Artificial Neural Network")
    print("Implementation from scratch using only NumPy")
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
    
    # Create neural network
    # Architecture: 8 input features -> 16 neurons -> 8 neurons -> 1 output
    print("Creating neural network...")
    layer_sizes = [X_train.shape[1], 16, 8, 1]
    print(f"Network architecture: {' -> '.join(map(str, layer_sizes))}")
    nn = NeuralNetwork(layer_sizes)
    print()
    
    # Train the model
    print("Training the model...")
    print("-" * 60)
    history = nn.train(
        X_train, y_train,
        X_val, y_val,
        epochs=500,
        learning_rate=0.01,
        batch_size=32,
        verbose=True
    )
    print("-" * 60)
    print()
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    metrics = evaluate_model(nn, X_test, y_test)
    
    print("\nTest Set Results:")
    print("=" * 60)
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
    predictions, probabilities = nn.predict(X_test[:10])
    for i in range(10):
        print(f"Sample {i+1}: Predicted: {predictions[i][0]}, "
              f"Probability: {probabilities[i][0]:.4f}, Actual: {int(y_test[i][0])}")
    print("-" * 60)


if __name__ == "__main__":
    main()