# Attention-LSTM Model with Hyperparameter Optimization

This repository contains a comprehensive implementation of attention mechanisms and LSTM blocks in TensorFlow, along with automated hyperparameter optimization using random search followed by grid search.

## Features

### Core Components

1. **AttentionBlock**: Multi-head attention mechanism with configurable parameters
   - Number of attention heads
   - Key dimension size
   - Dropout rates
   - Causal masking support
   - Layer normalization and feed-forward networks

2. **LSTMBlock**: Configurable LSTM block with multiple options
   - Configurable units and layers
   - Bidirectional support
   - Dropout and recurrent dropout
   - Layer normalization

3. **AttentionLSTMModel**: Functional model builder supporting multiple architectures
   - `attention_lstm`: Attention followed by LSTM
   - `lstm_attention`: LSTM followed by Attention
   - `parallel`: Parallel processing of both blocks
   - `stacked`: Multiple stacked layers of both blocks

4. **HyperparameterOptimizer**: Automated optimization pipeline
   - Random search for initial exploration
   - Grid search for fine-tuning around best parameters
   - Support for both regression and classification tasks

### Architecture Options

- **Sequential**: Attention → LSTM or LSTM → Attention
- **Parallel**: Both blocks process input simultaneously, outputs concatenated
- **Stacked**: Multiple layers alternating between attention and LSTM blocks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from attention_lstm_model import AttentionLSTMModel, HyperparameterOptimizer
import numpy as np

# Define input shape (sequence_length, features)
input_shape = (50, 10)

# Create model builder
model_builder = AttentionLSTMModel(input_shape)

# Build model with custom parameters
model = model_builder.build_model(
    attention_params={'num_heads': 8, 'key_dim': 64, 'dropout_rate': 0.1},
    lstm_params={'units': 128, 'dropout': 0.1, 'bidirectional': True},
    architecture='attention_lstm',
    num_classes=1,
    task_type='regression'
)

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)
```

### Hyperparameter Optimization

```python
# Initialize optimizer
optimizer = HyperparameterOptimizer(
    input_shape=(50, 10),
    task_type='regression',
    num_classes=1
)

# Perform random search
best_params = optimizer.random_search(
    X_train, y_train, X_val, y_val,
    n_iter=50,
    epochs=10
)

# Fine-tune with grid search
final_params = optimizer.grid_search(
    X_train, y_train, X_val, y_val,
    epochs=20
)

# Get optimized model
best_model, model_builder = optimizer.get_best_model()
```

### Complete Example

Run the complete pipeline with the provided example:

```bash
python example_usage.py
```

This will:
1. Generate synthetic time series data
2. Perform random search hyperparameter optimization
3. Fine-tune with grid search
4. Train the final model with best parameters
5. Evaluate on test set
6. Generate optimization plots
7. Compare different architectures

## Hyperparameter Search Space

The optimizer searches over the following parameters:

### Attention Parameters
- `num_heads`: [4, 8, 12, 16]
- `key_dim`: [32, 64, 128, 256]
- `dropout_rate`: [0.0, 0.1, 0.2, 0.3]

### LSTM Parameters
- `units`: [64, 128, 256, 512]
- `dropout`: [0.0, 0.1, 0.2, 0.3]
- `recurrent_dropout`: [0.0, 0.1, 0.2]
- `bidirectional`: [True, False]
- `num_layers`: [1, 2, 3]

### Architecture Options
- `architecture`: ['attention_lstm', 'lstm_attention', 'parallel', 'stacked']

### Training Parameters
- `batch_size`: [16, 32, 64, 128]
- `learning_rate`: [0.0001, 0.001, 0.01, 0.1]

## Model Architectures

### 1. Attention-LSTM (`attention_lstm`)
```
Input → Attention Block → LSTM Block → Dense Layers → Output
```

### 2. LSTM-Attention (`lstm_attention`)
```
Input → LSTM Block → Attention Block → Global Pooling → Dense Layers → Output
```

### 3. Parallel (`parallel`)
```
Input → Attention Block → Global Pooling ↘
                                           Concatenate → Dense Layers → Output
Input → LSTM Block ↗
```

### 4. Stacked (`stacked`)
```
Input → [Attention Block → LSTM Block] × N → Global Pooling → Dense Layers → Output
```

## Optimization Strategy

1. **Random Search**: Explores the hyperparameter space broadly to identify promising regions
2. **Grid Search**: Fine-tunes around the best parameters found in random search
3. **Early Stopping**: Prevents overfitting during parameter evaluation
4. **Memory Management**: Clears TensorFlow session after each evaluation to prevent memory leaks

## Customization

### Adding New Architectures

Extend the `build_model` method in `AttentionLSTMModel`:

```python
elif architecture == 'custom_arch':
    # Your custom architecture implementation
    x = custom_processing(inputs)
```

### Custom Search Spaces

Modify the `define_search_space` method in `HyperparameterOptimizer`:

```python
def define_search_space(self):
    search_space = {
        'custom_param': [value1, value2, value3],
        # ... other parameters
    }
    return search_space
```

## Performance Tips

1. **Memory Management**: The optimizer automatically clears TensorFlow sessions between evaluations
2. **Early Stopping**: Use early stopping during hyperparameter search to save time
3. **Batch Size**: Larger batch sizes generally train faster but require more memory
4. **Sequence Length**: Longer sequences require more memory, especially for attention mechanisms

## Output Files

- `best_attention_lstm_model.h5`: Saved model with best parameters
- `optimization_results.png`: Visualization of optimization progress

## Requirements

- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- Scikit-learn >= 1.0.0
- Matplotlib >= 3.5.0
- Seaborn >= 0.11.0