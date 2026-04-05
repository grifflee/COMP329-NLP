"""
=============================================================================
 MODEL 1: DENSE FEEDFORWARD NETWORK (Bag-of-Words Baseline)
=============================================================================

 ARCHITECTURE:
   Input → Embedding → GlobalAveragePooling1D → Dense → Dropout → Dense(softmax)

 KEY CONCEPTS INTRODUCED:
   ✦ Embedding Layer — converts each word (integer) into a dense vector
   ✦ GlobalAveragePooling1D — averages all word vectors into one vector
   ✦ Dense Layer — fully connected layer (every neuron connects to every input)
   ✦ Dropout — randomly "turns off" neurons during training to prevent overfitting
   ✦ Softmax — outputs probabilities that sum to 1 (one per genre)
   ✦ compile() / fit() / evaluate() — the Keras training loop

 WHY THIS MODEL?
   This is the SIMPLEST possible neural network for text classification.
   It treats each description as a "bag of words" — meaning word ORDER
   doesn't matter. "The dog bit the man" and "The man bit the dog" would
   produce the SAME prediction. Despite this limitation, it's surprisingly
   effective and establishes a baseline accuracy to beat.

 WHAT TO WATCH:
   - Training vs validation accuracy (gap = overfitting)
   - Which genres are easy vs hard to predict
   - Training time (should be fast, ~1 min)
=============================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ── Import our shared preprocessing ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.preprocessing import load_data, VOCAB_SIZE, MAX_LEN

# ── TensorFlow / Keras imports ──────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout


def build_model(num_classes):
    """
    Build the Dense Feedforward model.

    Architecture explanation:
    ┌──────────────────────────────────────────────────────────────────┐
    │ LAYER 1: Embedding(vocab_size=20000, embedding_dim=64)          │
    │                                                                  │
    │   Each word in our vocabulary is represented by an integer       │
    │   (e.g., "horror" → 532). The Embedding layer converts each     │
    │   integer into a 64-dimensional DENSE VECTOR.                   │
    │                                                                  │
    │   Input shape:  (batch_size, 200)      ← 200 word IDs           │
    │   Output shape: (batch_size, 200, 64)  ← 200 word vectors       │
    │                                                                  │
    │   These vectors are LEARNED during training — words with         │
    │   similar meanings end up with similar vectors.                  │
    ├──────────────────────────────────────────────────────────────────┤
    │ LAYER 2: GlobalAveragePooling1D()                                │
    │                                                                  │
    │   We have 200 word vectors, but we need ONE vector for the       │
    │   whole description. GlobalAveragePooling simply takes the       │
    │   AVERAGE of all 200 vectors.                                   │
    │                                                                  │
    │   Input shape:  (batch_size, 200, 64)  ← 200 word vectors       │
    │   Output shape: (batch_size, 64)       ← 1 average vector       │
    │                                                                  │
    │   ⚠️ This is where we LOSE word order information!               │
    ├──────────────────────────────────────────────────────────────────┤
    │ LAYER 3: Dense(64, activation='relu')                            │
    │                                                                  │
    │   A fully connected layer with 64 neurons. Each neuron           │
    │   computes: output = relu(W · input + b)                        │
    │                                                                  │
    │   ReLU activation: max(0, x) — keeps positive values,           │
    │   sets negative values to 0. This adds NON-LINEARITY            │
    │   (without it, stacking layers would be useless).               │
    ├──────────────────────────────────────────────────────────────────┤
    │ LAYER 4: Dropout(0.3)                                            │
    │                                                                  │
    │   During training, randomly sets 30% of neurons to 0.           │
    │   This prevents OVERFITTING — the model can't rely on any       │
    │   single neuron, forcing it to learn more robust features.      │
    │   During inference (prediction), all neurons are active.        │
    ├──────────────────────────────────────────────────────────────────┤
    │ LAYER 5: Dense(num_classes, activation='softmax')                │
    │                                                                  │
    │   Output layer with 27 neurons (one per genre).                 │
    │   Softmax converts raw scores into PROBABILITIES that sum to 1: │
    │     [2.1, 0.5, -1.0, ...] → [0.65, 0.13, 0.03, ...]           │
    │   The genre with the highest probability is the prediction.     │
    └──────────────────────────────────────────────────────────────────┘
    """
    model = Sequential([
        # Layer 1: Convert word IDs → dense vectors
        Embedding(
            input_dim=VOCAB_SIZE,        # Vocabulary size (20,000 words)
            output_dim=128,              # Each word becomes a 128-dim vector
        ),

        # Layer 2: Average all word vectors into one description vector
        GlobalAveragePooling1D(),

        # Layer 3: Hidden layer — learns patterns in the averaged vector
        Dense(128, activation='relu'),

        # Layer 4: Dropout — prevents overfitting
        Dropout(0.5),

        # Layer 5: Output — one probability per genre
        Dense(num_classes, activation='softmax')
    ])

    return model


def train_and_evaluate():
    """Main training pipeline."""

    # ── 1. Load preprocessed data ───────────────────────────────────────
    data = load_data()

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    num_classes = data['num_classes']
    class_weights = data['class_weights']
    genre_names = data['genre_names']

    # ── 2. Build the model ──────────────────────────────────────────────
    print("\n🏗️  Building Dense Feedforward model...")
    model = build_model(num_classes)

    # ── 3. Compile the model ────────────────────────────────────────────
    #
    # compile() configures the model for training:
    #
    #   optimizer='adam' — Adam is the go-to optimizer. It adapts the
    #     learning rate for each parameter automatically.
    #
    #   loss='sparse_categorical_crossentropy' — The loss function for
    #     multi-class classification where labels are integers (not one-hot).
    #     It measures how "wrong" the predictions are.
    #     "sparse" = labels are integers [0, 1, 2, ...] not one-hot [[1,0,0], [0,1,0], ...]
    #
    #   metrics=['accuracy'] — What we track during training (% correct).
    #
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print the model architecture
    model.summary()

    # ── 4. Train the model ──────────────────────────────────────────────
    #
    # fit() trains the model:
    #
    #   epochs=10 — Go through the ENTIRE training set 10 times.
    #     Each pass is called an "epoch". More epochs = more learning,
    #     but also more risk of overfitting.
    #
    #   batch_size=128 — Process 128 descriptions at a time before
    #     updating weights. Larger batches = faster but less precise updates.
    #
    #   validation_split=0.15 — Hold out 15% of training data for validation.
    #     This lets us monitor overfitting during training.
    #
    #   class_weight — Tells the model to weight errors by class frequency.
    #     Mistakes on rare genres (war) cost MORE than mistakes on common
    #     genres (drama). This prevents the model from just predicting "drama".
    #
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
    )

    print("\n🚀 Training Dense model...")
    history = model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=64,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # ── 5. Evaluate on test set ─────────────────────────────────────────
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Loss:     {test_loss:.4f}")

    # ── 6. Detailed classification report ───────────────────────────────
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n📋 Classification Report (per-genre breakdown):")
    print(classification_report(y_test, y_pred_classes, target_names=genre_names))

    # ── 7. Plot training curves ─────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model 1: Dense Feedforward — Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model 1: Dense Feedforward — Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '01_dense_results.png'), dpi=150)
    print("\n📈 Training curves saved to 01_dense_results.png")
    plt.show()

    # ── 8. Save metrics for comparison ──────────────────────────────────
    metrics = {
        'model': '01_Dense_Feedforward',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'epochs_trained': len(history.history['accuracy']),
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
    }
    metrics_path = os.path.join(os.path.dirname(__file__), '01_dense_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to 01_dense_metrics.json")

    return model, history


# ── Run the training ────────────────────────────────────────────────────────
if __name__ == '__main__':
    model, history = train_and_evaluate()
