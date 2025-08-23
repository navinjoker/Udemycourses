#!/usr/bin/env python3
"""
Example usage of the Attention-LSTM model with hyperparameter optimization
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from attention_lstm_model import AttentionLSTMModel, HyperparameterOptimizer
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_sample_data(n_samples=1000, sequence_length=50, n_features=10, task_type='regression'):
    """
    Generate sample time series data for demonstration
    """
    print(f"Generating sample data: {n_samples} samples, {sequence_length} timesteps, {n_features} features")
    
    # Generate synthetic time series data
    X = np.random.randn(n_samples, sequence_length, n_features)
    
    # Add some temporal patterns
    for i in range(n_samples):
        for j in range(n_features):
            # Add trend
            trend = np.linspace(0, np.random.randn(), sequence_length)
            # Add seasonality
            seasonal = np.sin(2 * np.pi * np.arange(sequence_length) / 10) * np.random.randn()
            # Add noise
            noise = np.random.randn(sequence_length) * 0.1
            
            X[i, :, j] += trend + seasonal + noise
    
    if task_type == 'regression':
        # For regression: predict next value based on sequence
        y = np.sum(X[:, -5:, :], axis=(1, 2)) + np.random.randn(n_samples) * 0.1
    else:  # classification
        # For classification: binary classification based on sequence properties
        y = (np.mean(X, axis=(1, 2)) > 0).astype(int)
    
    return X, y

def plot_optimization_results(optimizer):
    """
    Plot the optimization results
    """
    if not optimizer.results_history:
        print("No optimization results to plot")
        return
    
    # Extract data for plotting
    random_results = [r for r in optimizer.results_history if r['search_type'] == 'random']
    grid_results = [r for r in optimizer.results_history if r['search_type'] == 'grid']
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Optimization progress
    plt.subplot(1, 3, 1)
    if random_results:
        random_scores = [r['score'] for r in random_results]
        plt.plot(range(1, len(random_scores) + 1), random_scores, 'b-', label='Random Search', alpha=0.7)
    
    if grid_results:
        grid_scores = [r['score'] for r in grid_results]
        grid_start = len(random_results) + 1
        plt.plot(range(grid_start, grid_start + len(grid_scores)), grid_scores, 'r-', label='Grid Search', alpha=0.7)
    
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Hyperparameter Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Best scores over time
    plt.subplot(1, 3, 2)
    all_scores = [r['score'] for r in optimizer.results_history]
    if optimizer.task_type == 'regression':
        best_scores = np.minimum.accumulate(all_scores)
        plt.ylabel('Best MSE (lower is better)')
    else:
        best_scores = np.maximum.accumulate(all_scores)
        plt.ylabel('Best Accuracy (higher is better)')
    
    plt.plot(range(1, len(best_scores) + 1), best_scores, 'g-', linewidth=2)
    plt.xlabel('Iteration')
    plt.title('Best Score Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Architecture performance
    plt.subplot(1, 3, 3)
    arch_scores = {}
    for r in optimizer.results_history:
        arch = r['params']['architecture']
        if arch not in arch_scores:
            arch_scores[arch] = []
        arch_scores[arch].append(r['score'])
    
    architectures = list(arch_scores.keys())
    avg_scores = [np.mean(arch_scores[arch]) for arch in architectures]
    
    plt.bar(architectures, avg_scores)
    plt.xlabel('Architecture')
    plt.ylabel('Average Score')
    plt.title('Architecture Performance Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('/workspace/optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main execution function demonstrating the complete pipeline
    """
    print("=== Attention-LSTM Model with Hyperparameter Optimization ===\n")
    
    # Configuration
    SEQUENCE_LENGTH = 50
    N_FEATURES = 10
    N_SAMPLES = 2000
    TASK_TYPE = 'regression'  # Change to 'classification' for classification tasks
    NUM_CLASSES = 1
    
    # Generate sample data
    print("1. Generating sample data...")
    X, y = generate_sample_data(
        n_samples=N_SAMPLES,
        sequence_length=SEQUENCE_LENGTH,
        n_features=N_FEATURES,
        task_type=TASK_TYPE
    )
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples\n")
    
    # Initialize optimizer
    print("2. Initializing hyperparameter optimizer...")
    optimizer = HyperparameterOptimizer(
        input_shape=(SEQUENCE_LENGTH, N_FEATURES),
        task_type=TASK_TYPE,
        num_classes=NUM_CLASSES
    )
    
    # Perform random search
    print("3. Starting Random Search...")
    best_random_params = optimizer.random_search(
        X_train, y_train, X_val, y_val,
        n_iter=20,  # Reduced for demo purposes
        epochs=5,   # Reduced for demo purposes
        verbose=0
    )
    
    print(f"\nBest parameters from Random Search:")
    for key, value in best_random_params.items():
        print(f"  {key}: {value}")
    print(f"Best score: {optimizer.best_score:.4f}\n")
    
    # Perform grid search for fine-tuning
    print("4. Starting Grid Search for fine-tuning...")
    best_grid_params = optimizer.grid_search(
        X_train, y_train, X_val, y_val,
        epochs=10,  # Reduced for demo purposes
        verbose=0
    )
    
    print(f"\nBest parameters after Grid Search:")
    for key, value in best_grid_params.items():
        print(f"  {key}: {value}")
    print(f"Final best score: {optimizer.best_score:.4f}\n")
    
    # Build and train final model
    print("5. Building final model with best parameters...")
    final_model, model_builder = optimizer.get_best_model()
    
    print("Model architecture:")
    final_model.summary()
    
    # Train final model with more epochs
    print("\n6. Training final model...")
    history = final_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=best_grid_params['batch_size'],
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ]
    )
    
    # Evaluate on test set
    print("\n7. Evaluating on test set...")
    test_loss = final_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss[0]:.4f}")
    if len(test_loss) > 1:
        print(f"Test Metric: {test_loss[1]:.4f}")
    
    # Make predictions
    y_pred = final_model.predict(X_test)
    
    if TASK_TYPE == 'regression':
        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        mae = np.mean(np.abs(y_test - y_pred.flatten()))
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
    else:
        if NUM_CLASSES == 1:
            y_pred_binary = (y_pred > 0.5).astype(int)
            accuracy = np.mean(y_test == y_pred_binary.flatten())
        else:
            y_pred_classes = np.argmax(y_pred, axis=1)
            accuracy = np.mean(y_test == y_pred_classes)
        print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot results
    print("\n8. Generating optimization plots...")
    plot_optimization_results(optimizer)
    
    # Save final model
    final_model.save('/workspace/best_attention_lstm_model.h5')
    print("\nFinal model saved as 'best_attention_lstm_model.h5'")
    
    print("\n=== Pipeline completed successfully! ===")

def demo_different_architectures():
    """
    Demonstrate different architecture combinations
    """
    print("\n=== Demonstrating Different Architectures ===\n")
    
    # Generate small sample data for quick demo
    X, y = generate_sample_data(n_samples=500, sequence_length=30, n_features=8)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    architectures = ['attention_lstm', 'lstm_attention', 'parallel', 'stacked']
    results = {}
    
    for arch in architectures:
        print(f"Testing architecture: {arch}")
        
        model_builder = AttentionLSTMModel((30, 8))
        model = model_builder.build_model(
            attention_params={'num_heads': 4, 'key_dim': 64, 'dropout_rate': 0.1},
            lstm_params={'units': 128, 'dropout': 0.1, 'bidirectional': False},
            architecture=arch,
            num_classes=1,
            task_type='regression'
        )
        
        # Quick training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=5,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
        results[arch] = val_loss
        
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Parameters: {model.count_params():,}")
        print()
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
    
    print("Architecture Comparison Summary:")
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for arch, loss in sorted_results:
        print(f"  {arch}: {loss:.4f}")

if __name__ == "__main__":
    # Run main pipeline
    main()
    
    # Demonstrate different architectures
    demo_different_architectures()