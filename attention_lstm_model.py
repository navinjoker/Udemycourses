import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import ParameterSampler, ParameterGrid
from sklearn.metrics import mean_squared_error, accuracy_score
import itertools
import random
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AttentionBlock(layers.Layer):
    """
    Multi-head attention block with configurable parameters
    """
    def __init__(self, 
                 num_heads: int = 8, 
                 key_dim: int = 64,
                 dropout_rate: float = 0.1,
                 use_causal_mask: bool = False,
                 **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.use_causal_mask = use_causal_mask
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate
        )
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Feed-forward network
        self.ffn_dim = key_dim * 4
        self.dense1 = layers.Dense(self.ffn_dim, activation='relu')
        self.dense2 = layers.Dense(key_dim)
        
    def call(self, inputs, training=None, mask=None):
        # Multi-head attention
        attn_output = self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=mask,
            use_causal_mask=self.use_causal_mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.dense1(out1)
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout_rate': self.dropout_rate,
            'use_causal_mask': self.use_causal_mask
        })
        return config


class LSTMBlock(layers.Layer):
    """
    LSTM block with configurable parameters
    """
    def __init__(self, 
                 units: int = 128,
                 dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 return_sequences: bool = True,
                 bidirectional: bool = False,
                 num_layers: int = 1,
                 **kwargs):
        super(LSTMBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        
        self.lstm_layers = []
        for i in range(num_layers):
            return_seq = return_sequences if i == num_layers - 1 else True
            lstm_layer = layers.LSTM(
                units=units,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=return_seq,
                return_state=False
            )
            
            if bidirectional:
                lstm_layer = layers.Bidirectional(lstm_layer)
                
            self.lstm_layers.append(lstm_layer)
            
        self.layernorm = layers.LayerNormalization()
        
    def call(self, inputs, training=None, mask=None):
        x = inputs
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training, mask=mask)
        
        # Apply layer normalization if return_sequences is True
        if self.return_sequences:
            x = self.layernorm(x)
            
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'return_sequences': self.return_sequences,
            'bidirectional': self.bidirectional,
            'num_layers': self.num_layers
        })
        return config


class AttentionLSTMModel:
    """
    Functional model combining Attention and LSTM blocks
    """
    def __init__(self, input_shape: Tuple[int, int]):
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self, 
                   attention_params: Dict = None,
                   lstm_params: Dict = None,
                   architecture: str = 'attention_lstm',
                   num_classes: int = 1,
                   task_type: str = 'regression'):
        """
        Build model with specified architecture and parameters
        
        Args:
            attention_params: Parameters for attention block
            lstm_params: Parameters for LSTM block
            architecture: 'attention_lstm', 'lstm_attention', 'parallel', 'stacked'
            num_classes: Number of output classes
            task_type: 'regression' or 'classification'
        """
        if attention_params is None:
            attention_params = {'num_heads': 8, 'key_dim': 64, 'dropout_rate': 0.1}
        if lstm_params is None:
            lstm_params = {'units': 128, 'dropout': 0.1, 'return_sequences': True}
            
        # Input layer
        inputs = keras.Input(shape=self.input_shape)
        
        # Build architecture based on type
        if architecture == 'attention_lstm':
            # Attention first, then LSTM
            x = AttentionBlock(**attention_params)(inputs)
            lstm_params['return_sequences'] = False
            x = LSTMBlock(**lstm_params)(x)
            
        elif architecture == 'lstm_attention':
            # LSTM first, then Attention
            x = LSTMBlock(**lstm_params)(inputs)
            x = AttentionBlock(**attention_params)(x)
            x = layers.GlobalAveragePooling1D()(x)
            
        elif architecture == 'parallel':
            # Parallel processing
            attention_out = AttentionBlock(**attention_params)(inputs)
            attention_out = layers.GlobalAveragePooling1D()(attention_out)
            
            lstm_params['return_sequences'] = False
            lstm_out = LSTMBlock(**lstm_params)(inputs)
            
            x = layers.Concatenate()([attention_out, lstm_out])
            
        elif architecture == 'stacked':
            # Multiple layers of both
            x = inputs
            for i in range(2):  # 2 stacked layers
                x = AttentionBlock(**attention_params)(x)
                x = LSTMBlock(**{**lstm_params, 'return_sequences': True})(x)
            
            x = layers.GlobalAveragePooling1D()(x)
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Add dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        # Output layer
        if task_type == 'classification':
            if num_classes == 1:
                outputs = layers.Dense(1, activation='sigmoid')(x)
            else:
                outputs = layers.Dense(num_classes, activation='softmax')(x)
        else:  # regression
            outputs = layers.Dense(num_classes, activation='linear')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        if task_type == 'classification':
            if num_classes == 1:
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
            
        self.model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )
        
        return self.model
    
    def get_model_summary(self):
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Random Search and Grid Search
    """
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 task_type: str = 'regression',
                 num_classes: int = 1):
        self.input_shape = input_shape
        self.task_type = task_type
        self.num_classes = num_classes
        self.best_params = None
        self.best_score = float('inf') if task_type == 'regression' else 0.0
        self.results_history = []
        
    def define_search_space(self):
        """Define hyperparameter search space"""
        search_space = {
            # Attention parameters
            'attention_num_heads': [4, 8, 12, 16],
            'attention_key_dim': [32, 64, 128, 256],
            'attention_dropout': [0.0, 0.1, 0.2, 0.3],
            
            # LSTM parameters
            'lstm_units': [64, 128, 256, 512],
            'lstm_dropout': [0.0, 0.1, 0.2, 0.3],
            'lstm_recurrent_dropout': [0.0, 0.1, 0.2],
            'lstm_bidirectional': [True, False],
            'lstm_num_layers': [1, 2, 3],
            
            # Architecture
            'architecture': ['attention_lstm', 'lstm_attention', 'parallel', 'stacked'],
            
            # Training parameters
            'batch_size': [16, 32, 64, 128],
            'learning_rate': [0.001, 0.01, 0.1, 0.0001]
        }
        return search_space
    
    def random_search(self, 
                     X_train, y_train, 
                     X_val, y_val,
                     n_iter: int = 50,
                     epochs: int = 10,
                     verbose: int = 0):
        """
        Perform random search for hyperparameter optimization
        """
        print(f"Starting Random Search with {n_iter} iterations...")
        search_space = self.define_search_space()
        
        # Generate random parameter combinations
        param_list = list(ParameterSampler(search_space, n_iter=n_iter, random_state=42))
        
        for i, params in enumerate(param_list):
            print(f"Random Search Iteration {i+1}/{n_iter}")
            
            try:
                score = self._evaluate_params(params, X_train, y_train, X_val, y_val, epochs, verbose)
                
                # Update best parameters
                if self.task_type == 'regression':
                    if score < self.best_score:
                        self.best_score = score
                        self.best_params = params.copy()
                else:  # classification
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = params.copy()
                
                self.results_history.append({
                    'iteration': i+1,
                    'params': params,
                    'score': score,
                    'search_type': 'random'
                })
                
                print(f"Score: {score:.4f}, Best so far: {self.best_score:.4f}")
                
            except Exception as e:
                print(f"Error in iteration {i+1}: {str(e)}")
                continue
        
        print(f"Random Search completed. Best score: {self.best_score:.4f}")
        return self.best_params
    
    def grid_search(self, 
                   X_train, y_train, 
                   X_val, y_val,
                   base_params: Dict = None,
                   param_ranges: Dict = None,
                   epochs: int = 20,
                   verbose: int = 0):
        """
        Perform grid search around best parameters from random search
        """
        print("Starting Grid Search for fine-tuning...")
        
        if base_params is None:
            if self.best_params is None:
                raise ValueError("No base parameters provided. Run random_search first.")
            base_params = self.best_params.copy()
        
        # Define smaller ranges around best parameters
        if param_ranges is None:
            param_ranges = self._get_fine_tune_ranges(base_params)
        
        # Generate all parameter combinations
        param_grid = ParameterGrid(param_ranges)
        total_combinations = len(param_grid)
        
        print(f"Grid Search will evaluate {total_combinations} combinations")
        
        grid_best_score = self.best_score
        grid_best_params = base_params.copy()
        
        for i, param_updates in enumerate(param_grid):
            print(f"Grid Search Iteration {i+1}/{total_combinations}")
            
            # Merge base parameters with current grid parameters
            params = base_params.copy()
            params.update(param_updates)
            
            try:
                score = self._evaluate_params(params, X_train, y_train, X_val, y_val, epochs, verbose)
                
                # Update best parameters
                if self.task_type == 'regression':
                    if score < grid_best_score:
                        grid_best_score = score
                        grid_best_params = params.copy()
                else:  # classification
                    if score > grid_best_score:
                        grid_best_score = score
                        grid_best_params = params.copy()
                
                self.results_history.append({
                    'iteration': i+1,
                    'params': params,
                    'score': score,
                    'search_type': 'grid'
                })
                
                print(f"Score: {score:.4f}, Best so far: {grid_best_score:.4f}")
                
            except Exception as e:
                print(f"Error in iteration {i+1}: {str(e)}")
                continue
        
        # Update overall best parameters
        self.best_score = grid_best_score
        self.best_params = grid_best_params
        
        print(f"Grid Search completed. Best score: {self.best_score:.4f}")
        return self.best_params
    
    def _evaluate_params(self, params, X_train, y_train, X_val, y_val, epochs, verbose):
        """Evaluate a set of parameters"""
        # Build model with current parameters
        model_builder = AttentionLSTMModel(self.input_shape)
        
        attention_params = {
            'num_heads': params['attention_num_heads'],
            'key_dim': params['attention_key_dim'],
            'dropout_rate': params['attention_dropout']
        }
        
        lstm_params = {
            'units': params['lstm_units'],
            'dropout': params['lstm_dropout'],
            'recurrent_dropout': params['lstm_recurrent_dropout'],
            'bidirectional': params['lstm_bidirectional'],
            'num_layers': params['lstm_num_layers']
        }
        
        model = model_builder.build_model(
            attention_params=attention_params,
            lstm_params=lstm_params,
            architecture=params['architecture'],
            num_classes=self.num_classes,
            task_type=self.task_type
        )
        
        # Update learning rate
        model.optimizer.learning_rate = params['learning_rate']
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=params['batch_size'],
            verbose=verbose,
            callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
        )
        
        # Evaluate model
        y_pred = model.predict(X_val, verbose=0)
        
        if self.task_type == 'regression':
            score = mean_squared_error(y_val, y_pred)
        else:  # classification
            if self.num_classes == 1:
                y_pred_binary = (y_pred > 0.5).astype(int)
                score = accuracy_score(y_val, y_pred_binary)
            else:
                y_pred_classes = np.argmax(y_pred, axis=1)
                score = accuracy_score(y_val, y_pred_classes)
        
        # Clean up memory
        del model
        tf.keras.backend.clear_session()
        
        return score
    
    def _get_fine_tune_ranges(self, base_params):
        """Get fine-tuning ranges around best parameters"""
        ranges = {}
        
        # Attention parameters
        current_heads = base_params['attention_num_heads']
        ranges['attention_num_heads'] = [max(4, current_heads-4), current_heads, min(16, current_heads+4)]
        
        current_key_dim = base_params['attention_key_dim']
        ranges['attention_key_dim'] = [max(32, current_key_dim//2), current_key_dim, min(256, current_key_dim*2)]
        
        # LSTM parameters  
        current_units = base_params['lstm_units']
        ranges['lstm_units'] = [max(64, current_units//2), current_units, min(512, current_units*2)]
        
        # Learning rate fine-tuning
        current_lr = base_params['learning_rate']
        ranges['learning_rate'] = [current_lr/10, current_lr, current_lr*10]
        
        return ranges
    
    def get_best_model(self):
        """Build and return the best model found"""
        if self.best_params is None:
            raise ValueError("No optimization has been performed yet.")
        
        model_builder = AttentionLSTMModel(self.input_shape)
        
        attention_params = {
            'num_heads': self.best_params['attention_num_heads'],
            'key_dim': self.best_params['attention_key_dim'],
            'dropout_rate': self.best_params['attention_dropout']
        }
        
        lstm_params = {
            'units': self.best_params['lstm_units'],
            'dropout': self.best_params['lstm_dropout'],
            'recurrent_dropout': self.best_params['lstm_recurrent_dropout'],
            'bidirectional': self.best_params['lstm_bidirectional'],
            'num_layers': self.best_params['lstm_num_layers']
        }
        
        model = model_builder.build_model(
            attention_params=attention_params,
            lstm_params=lstm_params,
            architecture=self.best_params['architecture'],
            num_classes=self.num_classes,
            task_type=self.task_type
        )
        
        model.optimizer.learning_rate = self.best_params['learning_rate']
        
        return model, model_builder