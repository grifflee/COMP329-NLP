"""
=============================================================================
 MODEL 2: SIMPLE RNN (Recurrent Neural Network)
=============================================================================

 ARCHITECTURE:
   Input → Embedding → SimpleRNN → Dense → Dropout → Dense(softmax)

 KEY CONCEPTS INTRODUCED:
   ✦ SimpleRNN — processes tokens ONE BY ONE, maintaining a "hidden state"
   ✦ Hidden State — a vector that summarizes everything seen so far
   ✦ Sequential Processing — word ORDER now matters!

 HOW IT DIFFERS FROM MODEL 1 (Dense):
   Model 1 averaged all word vectors → lost word order.
   Model 2 reads words LEFT to RIGHT, updating its understanding at each step.

   This means "The dog bit the man" and "The man bit the dog" now produce
   DIFFERENT predictions (as they should).

 HOW AN RNN WORKS (step by step):
   ┌────────────────────────────────────────────────────────────────────┐
   │                                                                    │
   │  Input:  "A   young   detective   investigates   murders"          │
   │           ↓      ↓         ↓            ↓           ↓              │
   │          h₀ →  h₁  →    h₂    →      h₃     →    h₄              │
   │                                                    ↓               │
   │                                              final output          │
   │                                                                    │
   │  At each step:                                                     │
   │    h_t = tanh(W_input · x_t + W_hidden · h_{t-1} + bias)         │
   │                                                                    │
   │  The hidden state h_t depends on:                                  │
   │    1. The current word x_t                                         │
   │    2. The previous hidden state h_{t-1} (memory of past words)    │
   │                                                                    │
   └────────────────────────────────────────────────────────────────────┘

 LIMITATION:
   Vanilla RNNs struggle with LONG sequences due to the "vanishing gradient"
   problem. After ~20-30 words, earlier information starts to fade.
   This is why LSTM (Model 3) was invented.

 WHAT TO WATCH:
   - Does accuracy improve over Model 1? (probably slightly)
   - Training is SLOWER (sequential = can't parallelize)
   - The vanishing gradient issue may limit performance
=============================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.preprocessing import load_data, VOCAB_SIZE, MAX_LEN

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout


def build_model(num_classes):
    """
    Build the Simple RNN model.

    The key difference from Model 1 is replacing GlobalAveragePooling1D
    with a SimpleRNN layer. Instead of averaging all word vectors, the
    RNN reads them sequentially and produces a single output vector that
    captures the ORDER of words.
    """
    model = Sequential([
        # Layer 1: Word embeddings (same as Model 1)
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=64,
        ),

        # Layer 2: Simple RNN
        #   - Reads 200 word vectors one by one
        #   - Maintains a hidden state of size 64
        #   - return_sequences=False means only output the FINAL hidden state
        #     (the one after reading the last word)
        #   - If True, it would output a hidden state at EVERY step (200 outputs)
        SimpleRNN(
            units=64,                    # Hidden state size
            return_sequences=False,      # Only return the final state
            dropout=0.2,                 # Dropout on inputs (prevents overfitting)
            recurrent_dropout=0.2        # Dropout on hidden state connections
        ),

        # Layer 3: Hidden dense layer
        Dense(32, activation='relu'),

        # Layer 4: Dropout
        Dropout(0.3),

        # Layer 5: Output layer
        Dense(num_classes, activation='softmax')
    ])

    return model


def train_and_evaluate():
    """Main training pipeline."""

    data = load_data()
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    num_classes = data['num_classes']
    class_weights = data['class_weights']
    genre_names = data['genre_names']

    print("\n🏗️  Building Simple RNN model...")
    model = build_model(num_classes)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # RNNs are slower to train, so we use fewer epochs and a callback
    # to stop early if validation accuracy stops improving
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,                  # Stop if no improvement for 3 epochs
        restore_best_weights=True    # Keep the best model, not the last one
    )

    print("\n🚀 Training Simple RNN (this may take a few minutes)...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.15,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss:     {test_loss:.4f}")

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=genre_names))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model 2: Simple RNN — Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model 2: Simple RNN — Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '02_rnn_results.png'), dpi=150)
    print("\n📈 Saved to 02_rnn_results.png")
    plt.show()

    # Save metrics
    metrics = {
        'model': '02_Simple_RNN',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'epochs_trained': len(history.history['accuracy']),
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
    }
    with open(os.path.join(os.path.dirname(__file__), '02_rnn_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("💾 Metrics saved to 02_rnn_metrics.json")

    return model, history


if __name__ == '__main__':
    model, history = train_and_evaluate()
