"""
=============================================================================
 SHARED PREPROCESSING — Data Loading, Tokenization, and Padding
=============================================================================

This module is shared by ALL 5 neural network models so that results are
directly comparable. Every model calls:

    from shared.preprocessing import load_data
    data = load_data()

And gets back the same train/test split with the same tokenization.

KEY CONCEPTS:
  - Tokenization: Converting words → integer IDs
  - Padding: Making all sequences the same length
  - Label Encoding: Converting genre strings → integers
  - Class Weights: Handling imbalanced genres (drama=25% vs war=0.2%)
=============================================================================
"""

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# ─── Configuration ──────────────────────────────────────────────────────────
VOCAB_SIZE = 20_000    # Keep only the top 20,000 most frequent words
MAX_LEN = 200          # Truncate/pad all descriptions to 200 tokens
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_file(filepath, has_genre=True):
    """
    Parse the ::: delimited data files.

    Train format:  ID ::: TITLE ::: GENRE ::: DESCRIPTION
    Test format:   ID ::: TITLE ::: DESCRIPTION

    Args:
        filepath: Path to the .txt file
        has_genre: True for train/solution files, False for test files

    Returns:
        descriptions: list of description strings
        genres: list of genre strings (or None if has_genre=False)
    """
    descriptions = []
    genres = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # Split on the ::: delimiter
            parts = line.strip().split(' ::: ')

            if has_genre and len(parts) >= 4:
                # Train format: ID ::: TITLE ::: GENRE ::: DESCRIPTION
                genre = parts[2].strip()
                description = parts[3].strip()
                genres.append(genre)
                descriptions.append(description)

            elif not has_genre and len(parts) >= 3:
                # Test format: ID ::: TITLE ::: DESCRIPTION
                description = parts[2].strip()
                descriptions.append(description)

    if has_genre:
        return descriptions, genres
    else:
        return descriptions, None


def load_data(train_path=None, test_path=None):
    """
    Load and preprocess the IMDB genre classification dataset.

    This function:
      1. Parses the raw text files
      2. Tokenizes descriptions into integer sequences
      3. Pads sequences to a fixed length
      4. Encodes genre labels as integers
      5. Computes class weights for imbalanced genres

    Returns:
        dict with keys:
            X_train       - Padded training sequences, shape (N_train, MAX_LEN)
            X_test        - Padded test sequences, shape (N_test, MAX_LEN)
            y_train       - Integer labels for training, shape (N_train,)
            y_test        - Integer labels for test, shape (N_test,)
            num_classes   - Number of unique genres (27)
            class_weights - Dict mapping class index → weight (for imbalance)
            label_encoder - Fitted LabelEncoder (to convert back: index → genre)
            tokenizer     - Fitted Keras Tokenizer (to inspect vocabulary)
            genre_names   - List of genre names in label order
    """
    # We import TensorFlow/Keras here so the rest of the module can be
    # used for data inspection without requiring TF installed.
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # ── 1. Set default file paths ───────────────────────────────────────
    if train_path is None:
        train_path = os.path.join(DATA_DIR, 'train_data.txt')
    if test_path is None:
        test_path = os.path.join(DATA_DIR, 'test_data_solution.txt')

    print("=" * 60)
    print(" LOADING AND PREPROCESSING DATA")
    print("=" * 60)

    # ── 2. Parse raw files ──────────────────────────────────────────────
    print(f"\n📂 Reading training data from: {train_path}")
    train_descs, train_genres = parse_file(train_path, has_genre=True)
    print(f"   → {len(train_descs)} training samples loaded")

    print(f"\n📂 Reading test data from: {test_path}")
    test_descs, test_genres = parse_file(test_path, has_genre=True)
    print(f"   → {len(test_descs)} test samples loaded")

    # ── 3. Encode genre labels as integers ──────────────────────────────
    #
    # LabelEncoder converts strings to integers:
    #   "action" → 0, "adventure" → 1, "comedy" → 2, ...
    #
    # This is necessary because neural networks work with numbers,
    # not strings.
    print(f"\n🏷️  Encoding genre labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(train_genres)  # Learn label mapping from training data

    y_train = label_encoder.transform(train_genres)
    y_test = label_encoder.transform(test_genres)

    num_classes = len(label_encoder.classes_)
    genre_names = list(label_encoder.classes_)
    print(f"   → {num_classes} unique genres: {genre_names}")

    # ── 4. Tokenize text ────────────────────────────────────────────────
    #
    # The Tokenizer does two things:
    #   a) Builds a vocabulary: maps each word to a unique integer
    #      e.g., "the" → 1, "a" → 2, "man" → 45, ...
    #   b) Converts each description into a sequence of these integers
    #      e.g., "the man" → [1, 45]
    #
    # We keep only the top VOCAB_SIZE most frequent words. Any word
    # outside this vocabulary is replaced with an <OOV> (out-of-vocab) token.
    #
    print(f"\n📝 Tokenizing text (vocab_size={VOCAB_SIZE})...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_descs)  # Build vocabulary from training data ONLY

    # Convert text → sequences of integers
    X_train_seq = tokenizer.texts_to_sequences(train_descs)
    X_test_seq = tokenizer.texts_to_sequences(test_descs)

    # Show an example
    print(f"   → Vocabulary size: {min(len(tokenizer.word_index), VOCAB_SIZE)} words")
    print(f"   → Example: '{train_descs[0][:60]}...'")
    print(f"     becomes: {X_train_seq[0][:10]}...")

    # ── 5. Pad sequences ────────────────────────────────────────────────
    #
    # Neural networks need fixed-size inputs. But descriptions have
    # varying lengths (some are 10 words, others are 500+).
    #
    # Padding makes all sequences the same length:
    #   - Short sequences get zeros added to the END (post-padding)
    #   - Long sequences get TRUNCATED to MAX_LEN
    #
    # Example (MAX_LEN=5):
    #   [1, 45, 3]       → [1, 45, 3, 0, 0]      (padded)
    #   [1, 45, 3, 8, 2, 7, 9] → [1, 45, 3, 8, 2] (truncated)
    #
    print(f"\n📐 Padding sequences to length {MAX_LEN}...")
    X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

    print(f"   → X_train shape: {X_train.shape}")
    print(f"   → X_test shape:  {X_test.shape}")

    # ── 6. Compute class weights ────────────────────────────────────────
    #
    # Our dataset is IMBALANCED:
    #   drama = 25.1% (13,613 samples)
    #   war   =  0.2% (132 samples)
    #
    # Without class weights, the model would just predict "drama" for
    # everything and get 25% accuracy. Class weights tell the model to
    # pay MORE attention to rare genres and LESS to common ones.
    #
    # The weight for each class is inversely proportional to its frequency:
    #   weight = total_samples / (num_classes * class_count)
    #
    print(f"\n⚖️  Computing class weights for imbalanced genres...")
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(weights))

    # Show a few examples
    most_common_idx = np.argmin(weights)
    least_common_idx = np.argmax(weights)
    print(f"   → Most common:  '{genre_names[most_common_idx]}' → weight {weights[most_common_idx]:.3f}")
    print(f"   → Least common: '{genre_names[least_common_idx]}' → weight {weights[least_common_idx]:.3f}")

    print(f"\n{'=' * 60}")
    print(f" PREPROCESSING COMPLETE")
    print(f" Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f" Classes: {num_classes} | Vocab: {VOCAB_SIZE} | Seq length: {MAX_LEN}")
    print(f"{'=' * 60}\n")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'num_classes': num_classes,
        'class_weights': class_weights,
        'label_encoder': label_encoder,
        'tokenizer': tokenizer,
        'genre_names': genre_names,
    }


# ─── Quick test: run this file directly to inspect the data ────────────────
if __name__ == '__main__':
    data = load_data()
    print("\n🔍 Quick data inspection:")
    print(f"   X_train dtype: {data['X_train'].dtype}")
    print(f"   y_train dtype: {data['y_train'].dtype}")
    print(f"   First 5 labels: {data['y_train'][:5]}")
    print(f"   Corresponding genres: {[data['genre_names'][i] for i in data['y_train'][:5]]}")
