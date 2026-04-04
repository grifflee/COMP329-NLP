"""
=============================================================================
 MODEL 5: TRANSFORMER (Self-Attention)
=============================================================================

 ARCHITECTURE:
   Input → Embedding + PositionalEncoding → MultiHeadAttention →
   LayerNorm → FeedForward → GlobalAveragePooling → Dense(softmax)

 KEY CONCEPTS INTRODUCED:
   ✦ Self-Attention — every word "attends to" every other word
   ✦ Query, Key, Value — the three vectors that compute attention scores
   ✦ Multi-Head Attention — multiple "perspectives" of attention
   ✦ Positional Encoding — injecting word ORDER information
   ✦ Residual Connections — "skip connections" to help gradients flow
   ✦ Layer Normalization — stabilizes training

 WHY THIS IS THE MOST POWERFUL:

   RNN:  reads word by word — information must "travel" through each step
   CNN:  sees local windows — can't directly connect distant words
   TRANSFORMER:  EVERY word can directly attend to EVERY other word!

   Example: "The detective, who had been working on the case for twenty
            years, finally solved the mystery"

   RNN: By the time it reads "solved", it may have forgotten "detective"
   CNN: Can't connect "detective" and "solved" (too far apart for kernel)
   TRANSFORMER: "solved" directly attends to "detective" — instant connection!

 HOW SELF-ATTENTION WORKS:

   For each word, compute 3 vectors:
     Q (Query)  = "What am I looking for?"
     K (Key)    = "What do I contain?"
     V (Value)  = "What information do I provide?"

   Attention score = Q · K^T / √(d_k)
   → High score means "these two words are relevant to each other"

   Output = softmax(scores) · V
   → Weighted combination of values, based on relevance

   ┌──────────────────────────────────────────────────────────────────┐
   │                                                                  │
   │  "A   detective   solves   mysterious   murders"                 │
   │   ↕       ↕         ↕         ↕           ↕                     │
   │   Q₁      Q₂        Q₃        Q₄          Q₅  (Queries)        │
   │   K₁      K₂        K₃        K₄          K₅  (Keys)           │
   │   V₁      V₂        V₃        V₄          V₅  (Values)         │
   │                                                                  │
   │  For Q₃ ("solves"):                                              │
   │    score with K₁ ("A")         = 0.01  (low — not relevant)     │
   │    score with K₂ ("detective") = 0.45  (high — who solves?)     │
   │    score with K₃ ("solves")    = 0.10  (moderate — self)        │
   │    score with K₄ ("mysterious")= 0.14  (some relevance)        │
   │    score with K₅ ("murders")   = 0.30  (high — solves what?)   │
   │                                                                  │
   │  Output₃ = 0.01·V₁ + 0.45·V₂ + 0.10·V₃ + 0.14·V₄ + 0.30·V₅    │
   │  → A blend emphasizing "detective" and "murders"                │
   │                                                                  │
   └──────────────────────────────────────────────────────────────────┘

 WHY POSITIONAL ENCODING?

   Unlike RNNs, transformers process all words IN PARALLEL — they have
   NO inherent notion of word order. "Dog bites man" and "Man bites dog"
   would produce the SAME output without positional encoding.

   Positional encoding adds a unique pattern to each position:
     Position 0: [sin(0), cos(0), sin(0), cos(0), ...]
     Position 1: [sin(1/10000), cos(1/10000), ...]
     Position 2: [sin(2/10000), cos(2/10000), ...]

   These patterns let the model learn that position matters.

 WHAT TO WATCH:
   - Should be the most accurate model
   - Training may be slower due to attention complexity O(n²)
   - 200 tokens is manageable — transformers struggle with very long sequences
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add
)


# ─── Custom Layer: Positional Encoding ──────────────────────────────────────
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Adds positional information to the embedding vectors.

    Since transformers process all positions in parallel (no recurrence),
    they need explicit position information. This layer adds a unique
    sinusoidal pattern to each position.

    The formula uses sin/cos at different frequencies:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why sin/cos? Because the model can learn to attend to RELATIVE
    positions (position+k can be represented as a linear function of
    position for any fixed offset k).
    """

    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        # Pre-compute the positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)  # Even indices: sin
        pe[:, 1::2] = np.cos(position * div_term)  # Odd indices: cos

        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        # Add positional encoding to the input embeddings
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len, 'd_model': self.d_model})
        return config


# ─── Custom Layer: Transformer Block ───────────────────────────────────────
class TransformerBlock(tf.keras.layers.Layer):
    """
    One Transformer encoder block, consisting of:
      1. Multi-Head Self-Attention
      2. Add & Norm (residual connection + layer normalization)
      3. Feed-Forward Network (two Dense layers)
      4. Add & Norm

    A real Transformer (like BERT) stacks 12+ of these blocks.
    We use just 1 for simplicity and speed.
    """

    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Multi-Head Attention
        #   num_heads=4 means 4 separate attention "heads"
        #   Each head learns different attention patterns:
        #     Head 1 might focus on subject-verb relationships
        #     Head 2 might focus on adjective-noun relationships
        #     Head 3 might focus on genre-indicative keywords
        #     Head 4 might focus on something else entirely
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )

        # Feed-Forward Network (applied to each position independently)
        self.ffn_dense1 = Dense(ff_dim, activation='relu')
        self.ffn_dense2 = Dense(d_model)

        # Layer Normalization (stabilizes training)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # ── Self-Attention ──────────────────────────────────────────
        # query=key=value=inputs because it's SELF-attention
        # (each word attends to all other words in the same sequence)
        attn_output = self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)

        # ── Residual Connection + Layer Norm ────────────────────────
        # "Residual connection" means: add the INPUT back to the output
        # This helps gradients flow during training (skip connection)
        out1 = self.layernorm1(inputs + attn_output)

        # ── Feed-Forward Network ────────────────────────────────────
        ffn_output = self.ffn_dense1(out1)
        ffn_output = self.ffn_dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)

        # ── Another Residual + Layer Norm ───────────────────────────
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config


def build_model(num_classes):
    """
    Build the Transformer model using the Keras Functional API.

    We switch from Sequential to Functional API here because
    custom layers (TransformerBlock, PositionalEncoding) work
    more naturally with it.

    KEY FIX: The original version had d_model=64 which gave the
    attention mechanism very little capacity per head (64/4 = 16 dims).
    We bump to d_model=128 so each of the 4 heads gets 32 dimensions,
    and we scale embeddings by sqrt(d_model) as described in the
    original "Attention is All You Need" paper.
    """
    d_model = 128      # Embedding dimension (increased from 64)
    num_heads = 4      # Number of attention heads (32 dims/head now)
    ff_dim = 256       # Feed-forward network hidden size

    # ── Input ───────────────────────────────────────────────────────
    inputs = Input(shape=(MAX_LEN,))

    # ── Embedding + Scaling + Positional Encoding ───────────────────
    # The original Transformer paper scales embeddings by sqrt(d_model)
    # to keep the magnitude in a good range relative to positional encodings
    x = Embedding(input_dim=VOCAB_SIZE, output_dim=d_model)(inputs)
    x = x * tf.math.sqrt(tf.cast(d_model, tf.float32))  # Scale embeddings
    x = PositionalEncoding(max_len=MAX_LEN, d_model=d_model)(x)

    # ── Transformer Blocks (x2 for more capacity) ───────────────────
    # Stacking 2 blocks lets the model learn more complex patterns.
    # Real transformers like BERT use 12 blocks, GPT-3 uses 96.
    x = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout_rate=0.1
    )(x)
    x = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout_rate=0.1
    )(x)

    # ── Pooling ─────────────────────────────────────────────────────
    # Average all 200 position outputs into a single vector
    x = GlobalAveragePooling1D()(x)

    # ── Classification Head ─────────────────────────────────────────
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_and_evaluate():
    """Main training pipeline."""

    data = load_data()
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    num_classes = data['num_classes']
    class_weights = data['class_weights']
    genre_names = data['genre_names']

    print("\n🏗️  Building Transformer model...")
    model = build_model(num_classes)

    # FIX: Use a lower constant learning rate.
    # Transformers are notoriously sensitive to learning rate.
    # 5e-4 is a common starting point for small transformers.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,              # More patience — transformers warm up slowly
        restore_best_weights=True
    )

    print("\n🚀 Training Transformer (the most complex model)...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
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
    ax1.set_title('Model 5: Transformer — Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model 5: Transformer — Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '05_transformer_results.png'), dpi=150)
    print("\n📈 Saved to 05_transformer_results.png")
    plt.show()

    metrics = {
        'model': '05_Transformer',
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'epochs_trained': len(history.history['accuracy']),
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1]),
    }
    with open(os.path.join(os.path.dirname(__file__), '05_transformer_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("💾 Metrics saved to 05_transformer_metrics.json")

    return model, history


if __name__ == '__main__':
    model, history = train_and_evaluate()
