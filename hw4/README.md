# HW5

## How to Run

```bash
python3 hw5.py
```

## Required Files

The following data files must be in the same directory as hw5.py:
- `shakespeare.txt` - Shakespeare corpus for training a small Word2Vec model
- `combined.tab` - WordSim-353 dataset for Spearman correlation evaluation
- `rt-polarity.pos` - Positive movie review sentences
- `rt-polarity.neg` - Negative movie review sentences

## Dependencies

```bash
pip install gensim scipy numpy scikit-learn
```

## Output

The script runs five parts:
1. **Part 1** - Trains a Word2Vec model on the Shakespeare corpus and runs similarity queries
2. **Part 2** - Loads the pretrained Google News Word2Vec model and runs similarity queries on various word categories
3. **Part 3** - Evaluates the Google News model against the WordSim-353 dataset using Spearman's rank correlation
4. **Part 4** - Tests word analogies (e.g., dog - puppy + kitten = cat) using the Google News model
5. **Part 5** - Builds a sentiment classifier using averaged Word2Vec embeddings as features and evaluates on dev/test sets
