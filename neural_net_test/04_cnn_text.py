"""
=============================================================================
 MODEL 4: CNN FOR TEXT (Convolutional Neural Network)
=============================================================================

 ARCHITECTURE:
   Input → Embedding → Conv1D → GlobalMaxPooling1D → Dense → Dropout → Dense(softmax)

 KEY CONCEPTS INTRODUCED:
   ✦ Conv1D — a sliding "filter" that detects local word patterns (n-grams)
   ✦ Kernel Size — how many consecutive words the filter looks at
   ✦ Feature Maps (Filters) — each filter learns to detect a different pattern
   ✦ GlobalMaxPooling1D — picks the STRONGEST activation of each filter

 HOW CNNs WORK ON TEXT:

   Imagine sliding a window across the description:

   "A detective investigates mysterious murders in a small town"

   With kernel_size=3, the filter looks at 3 words at a time:
     Window 1: [A, detective, investigates]        → score 0.2
     Window 2: [detective, investigates, mysterious] → score 0.1
     Window 3: [investigates, mysterious, murders]   → score 0.8  ← crime pattern!
     Window 4: [mysterious, murders, in]             → score 0.7
     Window 5: [murders, in, a]                      → score 0.1
     ...

   GlobalMaxPooling picks the HIGHEST score (0.8) — the strongest match.

   With 128 different filters, the model learns 128 different patterns:
     Filter 1 might detect "romantic comedy about..."
     Filter 2 might detect "murders in a..."
     Filter 3 might detect "documentary about the..."
     ...

 WHY CNNs FOR TEXT?

   ┌─────────────────────────────────────┬──────────────────────────────────┐
   │            RNN/LSTM                 │              CNN                 │
   ├─────────────────────────────────────┼──────────────────────────────────┤
   │ Reads sequentially (slow)          │ All windows in parallel (fast)   │
   │ Good at long-range dependencies    │ Good at local patterns (n-grams) │
   │ "understands" word order deeply    │ Detects KEY PHRASES efficiently  │
   │ Can't parallelize across time      │ Highly parallelizable (GPU love) │
   └─────────────────────────────────────┴──────────────────────────────────┘

   For genre classification, LOCAL PATTERNS are often most informative:
     "horror" → "terrifying journey", "haunted house", "serial killer"
     "comedy" → "hilarious series", "laugh out loud", "wacky adventures"

 WHAT TO WATCH:
   - Training should be FASTER than LSTM
   - May match or exceed LSTM accuracy for this task
   - The kernel_size parameter controls n-gram size
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
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
)


def build_model(num_classes):
    """
    Build the CNN text classification model.

    Key idea: Conv1D filters act as n-gram detectors.
    A filter with kernel_size=5 is essentially learning to recognize
    important 5-word phrases in movie descriptions.
    """
    model = Sequential([
        # Layer 1: Word embeddings
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=64,
        ),

        # Layer 2: Conv1D — the star of this model
        #
        # filters=128: Learn 128 different pattern detectors
        # kernel_size=5: Each filter looks at 5 consecutive words
        # activation='relu': Only keep positive activations
        #
        # Input shape:  (batch, 200, 64)   — 200 word vectors of dim 64
        # Output shape: (batch, 196, 128)  — 196 positions × 128 filters
        #   (196 because a window of 5 can start at positions 0-195)
        #
        Conv1D(
            filters=128,             # Number of pattern detectors
            kernel_size=5,           # N-gram size (5-word window)
            activation='relu',
            padding='valid'          # No padding = output shrinks slightly
        ),

        # Layer 3: GlobalMaxPooling1D
        #
        # For each of the 128 filters, pick the MAXIMUM activation
        # across all 196 positions. This finds the STRONGEST match
        # for each pattern, regardless of WHERE in the text it appears.
        #
        # Input shape:  (batch, 196, 128)
        # Output shape: (batch, 128)
        #
        GlobalMaxPooling1D(),

        # Layer 4: Hidden dense layer
        Dense(64, activation='relu'),

        # Layer 5: Dropout
        Dropout(0.3),

        # Layer 6: Output
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

    print("\n🏗️  Building CNN Text Classification model...")
    model = build_model(num_classes)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )

    print("\n🚀 Training CNN (should be faster than LSTM!)...")
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
    ax1.set_title('Model 4: CNN for Text — Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model 4: CNN for Text — Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '04_cnn_results.png'), dpi=150)
    print("\n📈 Saved to 04_cnn_results.png")
    plt.show()

    metrics = {
        'model': '04_CNN_Text',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'epochs_trained': len(history.history['accuracy']),
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
    }
    with open(os.path.join(os.path.dirname(__file__), '04_cnn_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("💾 Metrics saved to 04_cnn_metrics.json")

    return model, history


if __name__ == '__main__':
    model, history = train_and_evaluate()
