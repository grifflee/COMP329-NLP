"""
=============================================================================
 MODEL 3: BIDIRECTIONAL LSTM (Long Short-Term Memory)
=============================================================================

 ARCHITECTURE:
   Input → Embedding → Bidirectional(LSTM) → Dense → Dropout → Dense(softmax)

 KEY CONCEPTS INTRODUCED:
   ✦ LSTM Cell — solves the vanishing gradient problem with GATES
   ✦ Cell State — a "conveyor belt" that carries important info across time
   ✦ Forget Gate — decides what to THROW AWAY from memory
   ✦ Input Gate — decides what NEW info to STORE in memory
   ✦ Output Gate — decides what to OUTPUT from memory
   ✦ Bidirectional — reads the description FORWARD and BACKWARD

 WHY LSTM OVER SIMPLE RNN?

   The Simple RNN (Model 2) has a problem: after reading 200 words,
   it has mostly "forgotten" what the first words were. This is the
   VANISHING GRADIENT problem — gradients shrink exponentially as they
   flow backward through time.

   LSTM solves this by adding a "cell state" — a separate memory track
   that can carry information across long distances without degradation.

   Think of it like a conveyor belt in a factory:
     - Regular RNN: information gets processed and degraded at every step
     - LSTM: important info rides the conveyor belt (cell state) unchanged,
             while gates control what gets added/removed

 LSTM CELL DIAGRAM:
   ┌─────────────────────────────────────────────────────────────────┐
   │                                                                 │
   │  Cell State (c_t) ──────────────────────────────── c_t          │
   │        ↑        ↑                         ↑                     │
   │     [FORGET]  [INPUT]                  [these are               │
   │      GATE      GATE                    pointwise                │
   │        │        │                      operations]              │
   │        │        │                                               │
   │   "What old  "What new                                         │
   │    info to    info to                                           │
   │    discard"   store"                                            │
   │                                                                 │
   │  Hidden State (h_t) ← [OUTPUT GATE] ← tanh(c_t)               │
   │                                                                 │
   │   "What to output as the result of this step"                  │
   │                                                                 │
   │  INPUTS: x_t (current word), h_{t-1} (prev hidden state)       │
   │                                                                 │
   └─────────────────────────────────────────────────────────────────┘

 BIDIRECTIONAL PROCESSING:
   Normal LSTM reads left→right. But sometimes the ENDING of a description
   is more informative than the beginning.

   Bidirectional wraps TWO LSTMs:
     Forward:  "A detective investigates a series of murders..."  →
     Backward: "...murders of series a investigates detective A"  ←

   Their outputs are concatenated, giving the model more context.

 WHAT TO WATCH:
   - Should handle long descriptions better than SimpleRNN
   - Output size DOUBLES because of Bidirectional (64×2 = 128)
   - Slower to train than SimpleRNN (but usually more accurate)
=============================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.preprocessing import load_data, VOCAB_SIZE, MAX_LEN

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, Dense, Dropout
)


def build_model(num_classes):
    """
    Build the Bidirectional LSTM model.

    Key differences from Model 2 (SimpleRNN):
    1. LSTM replaces SimpleRNN → gates allow long-range memory
    2. Bidirectional wrapper → reads forward AND backward
    3. Output is 128-dimensional (64 forward + 64 backward)
    """
    model = Sequential([
        # Layer 1: Word embeddings
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=128,
        ),

        # Layer 2: Bidirectional LSTM
        #
        # Bidirectional wraps TWO independent LSTM layers:
        #   - Forward LSTM: reads tokens 1, 2, 3, ..., 200
        #   - Backward LSTM: reads tokens 200, 199, ..., 1
        #
        # Their final hidden states are CONCATENATED:
        #   forward_output (64) + backward_output (64) = 128 dimensions
        #
        Bidirectional(
            LSTM(
                units=128,               # Hidden state size per direction
                dropout=0.3,             # Input dropout
            )
        ),
        # Output shape: (batch_size, 256) — because 128 × 2 directions

        # Layer 3: Hidden dense layer
        Dense(64, activation='relu'),

        # Layer 4: Dropout
        Dropout(0.5),

        # Layer 5: Output
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

    print("\n🏗️  Building Bidirectional LSTM model...")
    model = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-5
    )

    print("\n🚀 Training Bidirectional LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=64,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr],
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
    ax1.set_title('Model 3: Bidirectional LSTM — Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model 3: Bidirectional LSTM — Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '03_lstm_results.png'), dpi=150)
    print("\n📈 Saved to 03_lstm_results.png")
    plt.show()

    metrics = {
        'model': '03_Bidirectional_LSTM',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'epochs_trained': len(history.history['accuracy']),
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
    }
    with open(os.path.join(os.path.dirname(__file__), '03_lstm_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("💾 Metrics saved to 03_lstm_metrics.json")

    return model, history


if __name__ == '__main__':
    model, history = train_and_evaluate()
