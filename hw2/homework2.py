import math
import random

# 
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


# now I need to get the counts of everything to use for probabilities
# total reviews
total_reviews_in_training = len(training_file)


# total positive lines & negative lines in the training file
# initalize to 0
total_positive_reviews = 0
total_negative_reviews = 0

# get counts for both using the tuple labels
# gives total review numbers (# sentences)
for text, label in training_file:
    if label == "pos":
        total_positive_reviews +=1
    else: #negative
        total_negative_reviews +=1 


# counts number of words in a file, will call this multiple times
def word_counter(file):
    total = 0
    for text, label in file:
        # split sentences into words
        total += len(text.split())
    return total

# get the count of total words in the training file
total_words_in_training = word_counter(training_file)

# get the positive lines in the training file
positive_lines_in_training = []
negative_lines_in_training = []
for text, label in training_file:
    if label == 'pos':
        positive_lines_in_training.append((text, label))
    else:
        negative_lines_in_training.append((text, label))


# call the word counter on the positive lines
total_words_in_positive = word_counter(positive_lines_in_training)
# call the word counter function on the negative lines
total_words_in_negative = word_counter(negative_lines_in_training)

# going to make a function to set a dictionary for word frequencies
# key value, word freq

def calc_frequencies(data):
    word_frequencies = {}
    for text, label in data:
        words = text.split()
        for word in words:
            # if the word has been added already, increment
            if word in word_frequencies: 
                word_frequencies[word]+= 1
            #if the word hasn't been added yet, start the count at 1
            else: 
                word_frequencies[word] = 1
    return word_frequencies

# get the positive word frequency dictionary
positive_word_frequencies = calc_frequencies(positive_lines_in_training)
# get the negative word frequency dictionary
negative_word_frequencies = calc_frequencies(negative_lines_in_training)

# need to get the unique words so im going to use sets
# sets dont accept duplicates so this will shrink to unique counts
vocabulary = set()

for x in positive_word_frequencies:
    vocabulary.add(x)
for x in negative_word_frequencies:
    vocabulary.add(x)

# total # of unique words in training_file
vocabulary_size = len(vocabulary)

# probability pos review
prior_positive = total_positive_reviews / total_reviews_in_training
# probability neg review
prior_negative = total_negative_reviews /total_reviews_in_training

# calculating each word probability, going to add the + 1 and |V| smoothing from 
# slide 19 of the naive-bayes power point

# takes in the word, the word frequency count, the total words in the subsequent file, and total vocabulary size
def word_probability(word, word_frequency_dictionary, total_words_in_file, vocabulary_size):
    # init to 0
    word_count = 0
    # if the word has a frequency in the dict 
    if word in word_frequency_dictionary:
        # get the word count (value) otherwise its going to say 0
        word_count = word_frequency_dictionary[word]
        # returns count(w,c)+1 / count(c) + |V|
    return ((word_count + 1) / (total_words_in_file + vocabulary_size))

def bayes_classifier (movie_review):
    # take the sentence and break it into lower case words
    review_words = []
    review_words = movie_review.strip().lower().split()

    # switching to logs because my probabilities were zeroing out 
    # get the prior probabilities and putting them in the sym to start
    sum_positive_score = math.log(prior_positive)
    sum_negative_score = math.log(prior_negative)

    for word in review_words:

        #after testing the dev set, I thought I'd add this to see if it helps
        if word in {"and", "a", "but", "the", "for", "is", "or", "be", "in", "to", "of"}:
            continue
        # positive probability for each word added 
        # calling the word probability function, then converting to log and adding to sum 
        positive_prob = word_probability(word, positive_word_frequencies, total_words_in_positive, vocabulary_size)
        pos_log_prob = math.log(positive_prob)
        sum_positive_score += pos_log_prob

        # negative probability for each word added 
        # calling the word probability function, then converting to log and adding to sum 
        negative_prob = word_probability(word, negative_word_frequencies, total_words_in_negative, vocabulary_size)
        neg_log_prob = math.log(negative_prob)
        sum_negative_score += neg_log_prob

    # determine if movie review was positie or negative based on overall value 
    # closer to 0 is still higher prob so > 
    if sum_positive_score > sum_negative_score:
        return 'pos', sum_positive_score, sum_negative_score
    else:
        return 'neg' , sum_positive_score, sum_negative_score


# testing on the development file 
def test_classifier(file):
    correct = 0
    total = len(file)

    for text, label in file:
        prediction, ps, ns = bayes_classifier(text)
        if prediction == label:
            correct += 1

    accuracy = correct / total
    return accuracy
# accuracy is 77% which I think is fine

# need to manually change allllllllll the variables to the concatenation of training AND development
training_and_dev = training_file + development_file

# going to hardcode then line split i forgot to make an earlier function out of this
positive_lines_in_td = []
negative_lines_in_td =[]

for text, label in training_and_dev:
    if label == 'pos':
        positive_lines_in_td.append((text, label))
    else:
        negative_lines_in_td.append((text, label))

# get the frequencies 
td_positive_frequency = calc_frequencies(positive_lines_in_td)
td_negative_frequency = calc_frequencies(negative_lines_in_td)
# get the word counts for pos and negative

td_postive_count = word_counter(positive_lines_in_td)
td_negative_count = word_counter(negative_lines_in_td)







if __name__ == "__main__":
    # testing the classifier with just the dev data set
    print(f"Testing development file: Accuracy: {test_classifier(development_file)}")
        # accuracy was at 0.7736085053158224 before removing some of the common words
        # stopped counting some of the most common words and it went to 0.774859287054409
    # changing variables to run the concat of training and dev instead of just training
    # i should have made more functions earlier so that i didnt have to hard code this
    # but im running out of time now

    # change frequencies
    positive_word_frequencies = td_positive_frequency
    negative_word_frequencies = td_negative_frequency

    # change counts (total=)
    total_words_in_positive = td_postive_count
    total_words_in_negative = td_negative_count
    # change priors
    prior_positive = len(positive_lines_in_td) / len(training_and_dev)
    prior_negative = len(negative_lines_in_td) / len(training_and_dev)
    
    # update vocabulary
    vocabulary = set()
    for x in td_positive_frequency:
        vocabulary.add(x)
    for x in td_negative_frequency:
        vocabulary.add(x)
    vocabulary_size = len(vocabulary)

    print(f"Testing TEST file: Accuracy: {test_classifier(test_file)}")
    # still getting 77 with a .1 bump so I think thats fine

    # checking for classifier  or uncertain
    text, actual_label = test_file[5]
    prediction, pos_score, neg_score = bayes_classifier(text)
    print(f"Review:{text},actual: {actual_label}, prediction: {prediction}, Positive score: {pos_score}, Negative score: {neg_score} \n")

    # checking for classifier uncertain
    text, actual_label = test_file[26]
    prediction, pos_score, neg_score = bayes_classifier(text)
    print(f"Review:{text},actual: {actual_label}, prediction: {prediction}, Positive score: {pos_score}, Negative score: {neg_score} \n")

    # checking for classifier confident
    text, actual_label = test_file[52]
    prediction, pos_score, neg_score = bayes_classifier(text)
    print(f"Review:{text},actual: {actual_label}, prediction: {prediction}, Positive score: {pos_score}, Negative score: {neg_score} \n")

   # checking for classifier confident 
    text, actual_label = test_file[20]
    prediction, pos_score, neg_score = bayes_classifier(text)
    print(f"Review:{text},actual: {actual_label}, prediction: {prediction}, Positive score: {pos_score}, Negative score: {neg_score} \n")
