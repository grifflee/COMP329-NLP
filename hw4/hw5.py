from gensim.test.utils import datapath
from gensim import utils
from scipy.stats import spearmanr
import math
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


import gensim.models
import gensim.downloader as api
import numpy as np
import os


# part 1
class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = 'shakespeare.txt'
        # splitting on the conversation blocks per character instead of each line
        with open(corpus_path) as f: # shakespeare file from this folder
            text = f.read() # save as obj
        for block in text.split('\n\n'): #split at the end of character block the line of blank space separating
            block = block.strip() # delete extra char space
            if block: # true if theres data
                yield utils.simple_preprocess(block) # text preprocessing copied from gensim
        
# figured out a way to not retrain the model everytime i call it
if os.path.exists('shakespeare.model'):
    model = gensim.models.Word2Vec.load('shakespeare.model')

else: 
    character_text_blocks = MyCorpus()
    model = gensim.models.Word2Vec(sentences=character_text_blocks, min_count = 5)
    model.save('shakespeare.model')
print("part 1: \n")
print(f"Most similar words for 'king': {model.wv.most_similar("king")} \n ")
print(f"Most similar words for 'hate': {model.wv.most_similar("hate")} \n ")
print(f"Most similar words for 'rome': {model.wv.most_similar("rome")} \n ")
print(f"similarity between 'henry' and 'king': {model.wv.similarity("henry", "king")} \n")
print("#######################################################################################################")
###
# part 2

wv = api.load('word2vec-google-news-300')
print('part 2')
print(f"Most similar words for 'king': {wv.most_similar("king")} \n ")
print(f"Most similar words for 'football': {wv.most_similar("football")} \n ")
print(f"Most similar words for 'earthquake': {wv.most_similar("earthquake")} \n ")
print(f"similarity between 'gun' and 'republican': {wv.similarity("gun", "republican")} \n")
print(f"similarity between 'gun' and 'democrat': {wv.similarity("gun", "democrat")} \n")
print(f"Most similar words for 'chicago': {wv.most_similar("chicago")} \n ")
print(f"Most similar words for 'orange': {wv.most_similar("orange")} \n ")
print("#######################################################################################################")
###
#part 3

# lists to save the scores for each row/pair
wordsim_scores = []
googlenews_scores= []
# cant find the wordsim_similarity_goldstandard.txt so using combined.tab
with open('combined.tab') as f:
    next(f)
    # going through each row of data
    for line in f:
        # pull in data from file and append values based on order
        firstword, secondword, score = line.strip().split('\t')
        # adding the score col value for wordsim
        wordsim_scores.append(float(score))

        # based on the 2 words in word sim, calc the similarity score from googlenews
        googlenews_scores.append(wv.similarity(firstword, secondword))

    # using spearmanr calculation to compare the rankings of the two ordered arrays
    correlation, p_value = spearmanr(wordsim_scores, googlenews_scores)
print("part 3: \n")
print(f"the spearmanr correlation: {correlation} \n ")
print(f"the spearmanr p value {p_value} \n")
print("#######################################################################################################")
###

# part 4
print("part 4: \n")
# going back to the "large model" which im assuming is the google news agian

# print(f"word analogy: woman + old: {wv.most_similar(positive=["woman", "old"])} \n")
# print(f"word analogy: tall + building + city: {wv.most_similar(positive=["tall", "building", "city"])} \n")
# print(f"word analogy: water + ocean - salt: {wv.most_similar(positive=["water", "ocean"], negative=["salt"])} \n")
# print(f"word analogy: soil + seed + sun + water - pests: {wv.most_similar(positive=["soil", "seed", "sun", "water"], negative=["pests"])} \n")
print(f"word analogy: dog - puppy + kitten =  : {wv.most_similar(positive=["dog", "kitten"], negative=["puppy"])} \n")
print(f"word analogy: red - apple + banana = : {wv.most_similar(positive=["red", "banana"], negative=["apple"])} \n")
print(f"word analogy: movie - actor + musician = : {wv.most_similar(positive=["movie", "musician"], negative=["actor"])} \n")

print("part 5: \n")
#  Loading the positive and negative files and labelling each line so that
# when it gets mixed they don't lose their label
# calling strip (empty space) and lower (make sure upper and lower case gets merged)
with open('rt-polarity.pos', 'r', encoding='latin-1') as f:
    pos_data = [(line.strip().lower(), 'pos') for line in f if line.strip()]

with open('rt-polarity.neg', 'r', encoding='latin-1') as f:
    neg_data = [(line.strip().lower(), 'neg') for line in f if line.strip()]
# adding both lists to one big list 'total_data'
total_data = pos_data + neg_data

# randomize the reviews
random.seed(999)
random.shuffle(total_data)
# splitting into training first 70%, development next 15%, and test final 15%
# getting 70% of the data for training
# ints fix the float bug
training_border = int(len(total_data) * 0.70)
#getting next 15% chunk for development
# gets the border of the total data that adds up to 70 + 15
development_border = int(len(total_data) * 0.85)

test_border = int(len(total_data) * 1)

# first 70%
training_file = total_data[:training_border]
# next 15%
development_file= total_data[training_border:development_border]
# next 15%
test_file = total_data[development_border:test_border]


# new code to try and use the google news trained word embeddings 
def review_vector_embeddings(text, wv):
    words = text.split() # breaking review into individial words
    word_vectors = [wv[word] for word in words if word in wv] #if word has an embedded vector from google news w2v
    if word_vectors:
        return np.mean(word_vectors, axis=0) # avg vectors per review -> 1 vector
    else: 
        return np.zeros(300) # blank vector of zeros incase no data on word

# pull out the text and label per review, pass in the text (review line) and model vw from google news
train_matrix = np.array([review_vector_embeddings(text, wv) for text, label in training_file]) # review text
# 
train_labels = [label for text, label in training_file] # labels pulled by looping trainingfile
# for each text and label pair (row review) in the file get the labe l 

dev_matrix = np.array([review_vector_embeddings(text, wv) for text, label in development_file])
dev_labels = [label for text, label in development_file]

test_matrix = np.array([review_vector_embeddings(text, wv) for text, label in test_file])
test_labels = [label for text, label in test_file]

# run logistic regression and get the stats on dev and test
clf = LogisticRegression(C=1, max_iter=1000).fit(train_matrix, train_labels)
print(f"dev accuracy: {clf.score(dev_matrix, dev_labels)}")
print(f"test accuracy: {clf.score(test_matrix, test_labels)}")
