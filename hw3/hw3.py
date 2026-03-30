import math
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


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

# going to count the number of documents that a word appears in
def calc_num_docs_per_word(data):
    # create the empty dictionary
    review_counts = {}
    # loop over every sentence in the training file
    # tuple unpacking counts text as the sentence and label as the positive or neg
    for text, label in data:
        # for the current sentence, split into words, and get a list of only unique words (set removes dupes)
        unique_words = set(text.split())
        # inner loop runs for each word in the sentences unique additions
        for word in unique_words:
            # increment the document appearances for the word
            if word in review_counts: 
                review_counts[word]+= 1
            #if the word hasn't been added yet, start the count at 1
            else: 
                review_counts[word] = 1
    # return the dictionary that has each word and its number of appearances
    return review_counts

word_document_frequencies = calc_num_docs_per_word(training_file)
#print(word_document_frequencies)

# based on the number of documents a word appears in, it gets added (or not) to a cleaned list
def clean_dict(data):
    review_counts = {}
    index = 0
    # going over dictionary items
    for word, count in data.items():
        # hard coded to only add words that have a doc frequency of at least 50
        if count >= 1 and count <= 7000:
            # add the word at the specific index number (just iteration)
            review_counts[word] = index
            # go to next spot in the cleaned dictionary
            index += 1
    return review_counts

# creating the cleaned dictionary
cleaned_dictionary = clean_dict(word_document_frequencies)
#print(cleaned_dictionary)

# creating the vectors for each sentence based on the cleaned dictionary
# need to loop through the training file for each sentence

def vectorizer(file, dictionary):
    # empty list to store the vectors for each sentence, and the original sentence label
    list_of_labels = []
    list_of_vectors = []

    # unpack the sentence text and label (tuple) in training file
    for text, label in file:

        cur_vector = np.zeros(len(dictionary)) # create a vector for the current review and fill it with zeros at the determined format length
        # go through each word in the movie review
        for x in text.split():
            # for each word in the review check if its left in the cleaned dictionary
            if x in dictionary:
                # if the word is allowed to be a part of the vector then increment its count
                # the returned value of the cleaned dictionary access should match as the right placeholder
                index = dictionary[x]
                cur_vector[index] = cur_vector[index] + 1
        
       # keep the positive and negative label to use as a tag, indexing matches based on loop
        list_of_labels.append(label)
        # add the vector for this sentence / review
        list_of_vectors.append(cur_vector)
    # need to return the ENTIRE matrix of all vectors
    full_array = np.array(list_of_vectors)

    # return matrix and the labels
    return full_array, list_of_labels

matrix, labels = vectorizer(training_file, cleaned_dictionary)
#print(matrix)

#part3 training the logisticregression classifier

# C = "inverse of regularization strength, smaller values specify stronger regularization"labels
# max_iter = "maximum number of iterations taken for the solvers to converge"
# then for fit its just X and Y 
clf = LogisticRegression(C = 1, max_iter=1000).fit(matrix, labels)

# part 4
print("Manual vectorizer output: ")
dev_set_matrix, dev_set_labels = vectorizer(development_file, cleaned_dictionary)
predictions = clf.predict(dev_set_matrix)
accuracy = clf.score(dev_set_matrix, dev_set_labels)
print(f"dev set accuracy: {accuracy}")

test_matrix, test_labels = vectorizer(test_file, cleaned_dictionary)
predictions = clf.predict(test_matrix)
accuracy = clf.score(test_matrix, test_labels)
print(f"test set accuracy: {accuracy} \n")



###### part 5
# turn the tuples into lists of string
# this is for the training data
reviews = [text for text, label in training_file]
labels = [label for text, label in training_file]
# use the countVectorizer
count_vector = CountVectorizer()
training_matrix = count_vector.fit_transform(reviews)
clf_automatic = LogisticRegression(C=1.0, max_iter=1000).fit(training_matrix, labels)

# for the dev set I now need to strip the text down (AGAIN) and then reset the file arg
print("CountVectorizer vectorizer output: ")
dev_texts = [text for text, label in development_file]
dev_labels = [label for text, label in development_file]
dev_matrix = count_vector.transform(dev_texts)
print(f"dev set accuracy: {clf_automatic.score(dev_matrix, dev_labels)}")

# testing set
test_texts = [text for text, label in test_file]
test_labels = [label for text, label in test_file]
test_matrix = count_vector.transform(test_texts)
print(f"test set accuracy: {clf_automatic.score(test_matrix, test_labels)}\n")

# use the tfidfvectorizer
tf_vector = TfidfVectorizer()
training_matrix = tf_vector.fit_transform(reviews)
clf_automatic_tf = LogisticRegression(C=1.0, max_iter=1000).fit(training_matrix, labels)

# for the dev set I now need to strip the text down (AGAIN) and then reset the file arg
print("Tfidfvectorizer vectorizer output: ")
dev_texts = [text for text, label in development_file]
dev_labels = [label for text, label in development_file]
dev_matrix = tf_vector.transform(dev_texts)
print(f"dev set accuracy: {clf_automatic_tf.score(dev_matrix, dev_labels)}")

# testing set
test_texts = [text for text, label in test_file]
test_labels = [label for text, label in test_file]
test_matrix = tf_vector.transform(test_texts)
print(f"test set accuracy: {clf_automatic_tf.score(test_matrix, test_labels)}")
