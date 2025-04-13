import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import time

# Data preparation functions
def prepare_data(dataset_path, task_type='classification', test_size=0.2, val_size=0.2, random_state=42):
    """
    Prepare and preprocess data for model training and evaluation.
    """
    # Load data
    df = pd.read_csv(dataset_path)
    
    # Assume the last column is the target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Split data into train and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Split train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Process target based on task type
    if task_type == 'classification':
        # Check if binary or multi-class
        unique_classes = np.unique(y_train)
        if len(unique_classes) > 2:
            # One-hot encode for multi-class
            y_train = keras.utils.to_categorical(y_train)
            y_val = keras.utils.to_categorical(y_val)
            y_test = keras.utils.to_categorical(y_test)
            
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# Traditional ANN for comparison
def create_traditional_ann(input_shape, task_type='classification', num_classes=None):
    """
    Create a traditional Artificial Neural Network for comparison.
    """
    inputs = keras.layers.Input(shape=input_shape)
    
    # Hidden layers
    x = keras.layers.Dense(128, activation='relu')(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    
    # Output layer based on task type
    if task_type == 'classification':
        if num_classes is None or num_classes == 2:  # Binary classification
            outputs = keras.layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:  # Multi-class classification
            outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
    else:  # Regression
        outputs = keras.layers.Dense(1, activation='linear')(x)
        loss = 'mean_squared_error'
        metrics = ['mae']
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=metrics
    )
    
    return model

class EquivariantLayer(keras.layers.Layer):
    """
    A simplified implementation of equivariant transformations
    Inspired by the HERMES architecture
    """
    def __init__(self, units, activation='relu', **kwargs):
        super(EquivariantLayer, self).__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        
    def build(self, input_shape):
        # Main transformation weights
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='kernel'
        )
        # Equivariant bias
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias'
        )
        # Transformation matrix for equivariance
        self.transform = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            name='transform'
        )
        self.built = True
    
    def call(self, inputs):
        # Main transformation
        output = tf.matmul(inputs, self.kernel) + self.bias
        
        # Apply equivariant transformation
        eq_output = tf.matmul(inputs, self.transform)
        output = output + eq_output
        
        # Apply activation
        if self.activation_name:
            output = tf.keras.activations.get(self.activation_name)(output)
            
        return output

class SelfAttentionLayer(keras.layers.Layer):
    """
    Self-attention mechanism for the HyperFabric Neural Network
    """
    def __init__(self, attention_dim, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
    
    def build(self, input_shape):
        self.query = self.add_weight(
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            name='query'
        )
        self.key = self.add_weight(
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            name='key'
        )
        self.value = self.add_weight(
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            name='value'
        )
        self.built = True
    
    def call(self, inputs):
        # Calculate attention
        q = tf.matmul(inputs, self.query)
        k = tf.matmul(inputs, self.key)
        v = tf.matmul(inputs, self.value)
        
        # Scaled dot-product attention
        attention_weights = tf.matmul(q, k, transpose_b=True)
        attention_weights = attention_weights / tf.sqrt(tf.cast(self.attention_dim, dtype=tf.float32))
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        
        # Apply attention to values
        attended = tf.matmul(attention_weights, v)
        
        return attended

class CapsuleLayer(keras.layers.Layer):
    """
    Simplified Capsule Layer for the HyperFabric Neural Network
    This implementation avoids complex dynamic routing to prevent shape issues
    """
    def __init__(self, num_capsules, dim_capsules, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        
    def build(self, input_shape):
        # Primary transformation
        self.W = self.add_weight(
            shape=(input_shape[2], self.num_capsules * self.dim_capsules),
            initializer='glorot_uniform',
            name='W'
        )
        
        # Secondary transformation for capsule properties
        self.V = self.add_weight(
            shape=(self.num_capsules, input_shape[2], self.dim_capsules),
            initializer='glorot_uniform',
            name='V'
        )
        
        self.built = True
    
    def call(self, inputs):
        # Get batch size
        batch_size = tf.shape(inputs)[0]
        
        # First transformation with primary capsules
        # Shape: [batch, input_capsules, num_capsules * dim_capsules]
        transformed = tf.matmul(inputs, self.W)
        
        # Reshape to separate capsules
        # Shape: [batch, input_capsules, num_capsules, dim_capsules]
        reshaped = tf.reshape(transformed, [-1, tf.shape(inputs)[1], self.num_capsules, self.dim_capsules])
        
        # Aggregate across input capsules
        # Shape: [batch, num_capsules, dim_capsules]
        outputs = tf.reduce_mean(reshaped, axis=1)
        
        # Apply non-linearity (squashing)
        squared_norm = tf.reduce_sum(tf.square(outputs), axis=-1, keepdims=True)
        scale = squared_norm / (1.0 + squared_norm) / tf.sqrt(squared_norm + 1e-8)
        outputs = scale * outputs
        
        return outputs
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_capsules, self.dim_capsules)

class HyperFabricBlock(keras.layers.Layer):
    """
    Core building block for the HyperFabric Neural Network
    """ 
    def __init__(self, units, num_paths=3, dropout_rate=0.3, **kwargs):
        super(HyperFabricBlock, self).__init__(**kwargs)
        self.units = units
        self.num_paths = num_paths
        self.dropout_rate = dropout_rate
        
        # Define layers as class attributes
        self.dense_layers = []
        self.eq_layers = []
        self.dropout_layers = []
        self.bn_layers = []
        
    def build(self, input_shape):
        # Create path layers
        for i in range(self.num_paths):
            if i % 3 == 0:
                self.eq_layers.append(EquivariantLayer(self.units, activation='relu'))
            elif i % 3 == 1:
                self.dense_layers.append(keras.layers.Dense(self.units, activation='tanh'))
            else:
                self.dense_layers.append(keras.layers.Dense(self.units, activation='sigmoid'))
            
            if i % 2 == 0:
                self.dropout_layers.append(keras.layers.Dropout(self.dropout_rate))
            else:
                self.bn_layers.append(keras.layers.BatchNormalization())
        
        # Self-attention layer for path integration
        self.attention = SelfAttentionLayer(self.units)
        
        # Skip connection handling
        self.skip_connection = None
        if input_shape[-1] != self.units:
            self.skip_connection = keras.layers.Dense(self.units, use_bias=False)
        
        self.built = True
    
    def call(self, inputs, training=None):
        # Process inputs through each path
        path_outputs = []
        
        for i in range(self.num_paths):
            x = inputs
            
            # First layer in path
            if i % 3 == 0:
                x = self.eq_layers[i//3](x)
            else:
                x = self.dense_layers[i - (i//3) - 1](x)
            
            # Regularization layer
            if i % 2 == 0:
                x = self.dropout_layers[i//2](x, training=training)
            else:
                x = self.bn_layers[i//2](x, training=training)
            
            path_outputs.append(x)
        
        # Stack path outputs
        # Shape: [batch_size, num_paths, units]
        stacked = tf.stack(path_outputs, axis=1)
        
        # Apply self-attention for feature integration
        attended = self.attention(stacked)
        
        # Pool across path dimension
        output = tf.reduce_mean(attended, axis=1)
        
        # Apply skip connection
        if self.skip_connection is not None:
            output = output + self.skip_connection(inputs)
        else:
            if inputs.shape[-1] == self.units:
                output = output + inputs
        
        return output

def create_hyperfabric_nn(input_shape, task_type='classification', num_classes=None, complexity=3):
    """
    Create a HyperFabric Neural Network
    """
    inputs = keras.layers.Input(shape=input_shape)
    
    # Scale complexity parameters based on desired complexity level
    num_blocks = complexity
    fabric_width = 64 * complexity
    num_paths = 2 + complexity // 2
    
    # Initial embedding
    x = keras.layers.Dense(fabric_width, activation='relu')(inputs)
    
    # HyperFabric blocks
    for i in range(num_blocks):
        units = fabric_width // (2 ** min(i, 2))
        x = HyperFabricBlock(units, num_paths=num_paths, dropout_rate=0.3)(x)
    
    # Feature integration with capsules for higher-level reasoning
    # Reshape for capsule layer
    reshaped = keras.layers.Reshape((1, x.shape[-1]))(x)
    x = CapsuleLayer(num_capsules=8, dim_capsules=16)(reshaped)
    x = keras.layers.Flatten()(x)
    
    # Final representation
    x = keras.layers.Dense(32, activation='relu')(x)
    
    # Output layer based on task type
    if task_type == 'classification':
        if num_classes is None or num_classes == 2:  # Binary classification
            outputs = keras.layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:  # Multi-class classification
            outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
    else:  # Regression
        outputs = keras.layers.Dense(1, activation='linear')(x)
        loss = 'mean_squared_error'
        metrics = ['mae']
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=metrics
    )
    
    return model

# Model evaluation function
def train_and_evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                             task_type='classification', batch_size=32, epochs=50):
    """
    Train and evaluate a model.
    """
    results = {}
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Evaluate model
    inference_start_time = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - inference_start_time
    
    # Calculate metrics based on task type
    if task_type == 'classification':
        # Process predictions
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:  # Multi-class
            y_pred = np.argmax(predictions, axis=1)
            if len(y_test.shape) > 1:  # If y_test is one-hot encoded
                y_true = np.argmax(y_test, axis=1)
            else:
                y_true = y_test
        else:  # Binary
            y_pred = (predictions > 0.5).astype(int).flatten()
            y_true = y_test
        
        # Calculate metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, average='weighted')
        results['recall'] = recall_score(y_true, y_pred, average='weighted')
        results['f1'] = f1_score(y_true, y_pred, average='weighted')
    
    else:  # Regression
        y_pred = predictions.flatten()
        
        # Calculate metrics
        results['mae'] = mean_absolute_error(y_test, y_pred)
        results['mse'] = mean_squared_error(y_test, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['r2'] = r2_score(y_test, y_pred)
    
    # Store training history and time
    results['history'] = history.history
    results['training_time'] = training_time
    results['inference_time'] = inference_time
    results['inference_time_per_sample'] = inference_time / len(X_test)
    results['model_params'] = model.count_params()
    
    print(f"\n{model_name} Results:")
    for key, value in results.items():
        if key != 'history':
            print(f"{key}: {value}")
    
    return results

def train_and_evaluate_hyperfabric(X_train, y_train, X_val, y_val, X_test, y_test, 
                                  task_type='classification', batch_size=32, epochs=50,
                                  complexity=3):
    """
    Train and evaluate a HyperFabric Neural Network
    """
    # Determine input shape and number of classes
    input_shape = (X_train.shape[1],)
    
    if task_type == 'classification':
        if len(y_train.shape) > 1:  # One-hot encoded
            num_classes = y_train.shape[1]
        else:
            num_classes = len(np.unique(y_train))
            # If binary classification, set to None
            if num_classes == 2:
                num_classes = None
    else:
        num_classes = None
    
    # Create a HyperFabric Neural Network
    model = create_hyperfabric_nn(input_shape, task_type, num_classes, complexity)
    
    # Use the common evaluation function
    return train_and_evaluate_model(
        model=model,
        model_name="HyperFabric Neural Network",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        task_type=task_type,
        batch_size=batch_size,
        epochs=epochs
    )

def run_comprehensive_comparison(dataset_path, task_type='classification', batch_size=32, epochs=50, hfnn_complexity=3):
    """
    Run a comprehensive comparison between Traditional ANN and HyperFabric Neural Network
    """
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(dataset_path, task_type)
    
    # Get input shape
    input_shape = (X_train.shape[1],)
    
    # Get number of classes for classification
    if task_type == 'classification':
        if len(y_train.shape) > 1:  # One-hot encoded
            num_classes = y_train.shape[1]
        else:
            num_classes = len(np.unique(y_train))
            # If binary classification, set to None
            if num_classes == 2:
                num_classes = None
    else:
        num_classes = None
    
    # Create and train traditional ANN
    traditional_ann = create_traditional_ann(input_shape, task_type, num_classes)
    traditional_results = train_and_evaluate_model(
        traditional_ann, "Traditional ANN", 
        X_train, y_train, X_val, y_val, X_test, y_test,
        task_type, batch_size, epochs
    )
    
    # Train and evaluate HyperFabric Neural Network
    hyperfabric_results = train_and_evaluate_hyperfabric(
        X_train, y_train, X_val, y_val, X_test, y_test,
        task_type, batch_size, epochs, hfnn_complexity
    )
    
    return {
        'Traditional ANN': traditional_results,
        'HyperFabric Neural Network': hyperfabric_results,
        'task_type': task_type
    }

def visualize_results(results):
    """
    Visualize the comparison results between models
    """
    task_type = results['task_type']
    
    # Create comparison table
    if task_type == 'classification':
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    else:  # Regression
        metrics = ['mae', 'mse', 'rmse', 'r2']
        metric_names = ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error', 'R² Score']
    
    # Print comparison
    print("\nPERFORMANCE METRICS COMPARISON:")
    print("-" * 60)
    print(f"{'Metric':<30} | {'Traditional ANN':<25} | {'HyperFabric Neural Network':<25} | {'Improvement':<15}")
    print("-" * 100)
    
    for metric, name in zip(metrics, metric_names):
        ann_value = results['Traditional ANN'][metric]
        hfnn_value = results['HyperFabric Neural Network'][metric]
        
        # Calculate percentage improvement
        if metric in ['mae', 'mse', 'rmse']:  # Lower is better
            improvement = (ann_value - hfnn_value) / ann_value * 100
            direction = '↓'
        else:  # Higher is better
            improvement = (hfnn_value - ann_value) / ann_value * 100
            direction = '↑'
        
        print(f"{name:<30} | {ann_value:<25.6f} | {hfnn_value:<25.6f} | {improvement:<10.2f}% {direction}")
    
    # Model efficiency comparison
    ann_params = results['Traditional ANN']['model_params']
    hfnn_params = results['HyperFabric Neural Network']['model_params']
    
    ann_inference = results['Traditional ANN']['inference_time_per_sample'] * 1000  # ms
    hfnn_inference = results['HyperFabric Neural Network']['inference_time_per_sample'] * 1000  # ms
    
    print("\nCOMPUTATIONAL TRADE-OFFS:")
    print(f"- HyperFabric Neural Network has {hfnn_params/ann_params:.2f}x the parameters of Traditional ANN")
    print(f"- HyperFabric Neural Network inference time is {hfnn_inference/ann_inference:.2f}x that of Traditional ANN")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Loss During Training')
    plt.plot(results['Traditional ANN']['history']['loss'], label='Traditional ANN (Train)')
    plt.plot(results['Traditional ANN']['history']['val_loss'], label='Traditional ANN (Val)')
    plt.plot(results['HyperFabric Neural Network']['history']['loss'], label='HyperFabric (Train)')
    plt.plot(results['HyperFabric Neural Network']['history']['val_loss'], label='HyperFabric (Val)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    metric_key = 'accuracy' if task_type == 'classification' else 'mae'
    metric_name = 'Accuracy' if task_type == 'classification' else 'Mean Absolute Error'
    
    plt.title(f'{metric_name} During Training')
    plt.plot(results['Traditional ANN']['history'][metric_key], label='Traditional ANN (Train)')
    plt.plot(results['Traditional ANN']['history']['val_' + metric_key], label='Traditional ANN (Val)')
    plt.plot(results['HyperFabric Neural Network']['history'][metric_key], label='HyperFabric (Train)')
    plt.plot(results['HyperFabric Neural Network']['history']['val_' + metric_key], label='HyperFabric (Val)')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
def example_usage():
    """
    Example of how to use the HyperFabric Neural Network
    """
    # Classification example using Breast Cancer dataset
    from sklearn.datasets import load_breast_cancer
    
    # Load and prepare data
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    
    # Create a temporary CSV file
    import pandas as pd
    df = pd.DataFrame(np.c_[X, y], columns=list(cancer.feature_names) + ['target'])
    temp_file = "temp_breast_cancer.csv"
    df.to_csv(temp_file, index=False)
    
    # Run comparison
    results = run_comprehensive_comparison(
        dataset_path=temp_file,
        task_type='classification',
        batch_size=32,
        epochs=30,
        hfnn_complexity=2
    )
    
    # Visualize results
    visualize_results(results)
    
    # Clean up
    import os
    os.remove(temp_file)
    
    return results

# Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run example
    results = example_usage()
