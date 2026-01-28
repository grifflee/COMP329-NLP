
# function to load the review files
# each film review line should load as an element 
def load_files(x):
    with open(x, 'r') as file:
        film_reviews = file.readlines()
     # list with 
    film_reviews = [review.strip() for review in film_reviews]
    return film_reviews

# calling the load_files function on the two files
negative_reviews = load_files("rt-polarity.neg")
positive_reviews = load_files("rt-polarity.pos")
    
#print(negative_reviews)
#print(positive_reviews)
#print(len(negative_reviews))

#tried hardcoding the words and realized that was a mistake
''' 
list_of_positive_words = ["great", "fresh", "superb", "excellent", "charisma", "talent", "talented", "solid", "rich", "satisfying",
"nice", "good", "intelligent", "passion", "fun", "engrossing", "best", "memorable", "greatness", "beyond", "hilarious", "insightful", 
"improvement", "delivers", "worth", "precious", "treat", "savory", "satisfies", "elevates", "heart-warming", "impressed", "impressive",
"more", "amazing", "soar", "gem", "hidden", "despite", "beautiful", "beautifully", "well", "light" ]

list_of_negative_words = ["would've", "reeks", "empty", "forgot", "obvious", "hastily", "bad", "horrible", "bogging", "superficial",
"poor", "overinflated", "ugly", "weird", "little", "sluggish", "slow", "mindless", "junk", "waste", ""]
''' 


# function to calculate the top 200 words in each file
def word_counts(file):

    count_for_word = {}

    for review in file:
        # need to call split otherwise its just doing the chars
        for word in review.split():
            if word in count_for_word:
                count_for_word[word] += 1
            else: 
                count_for_word[word] = 1

    #syntax to sort the dictionary by largest to smallest value by making tuple list
    sorted_dictionary = sorted(count_for_word.items(), key=lambda x: x[1], reverse=True)

    popular_words = []
    for x in sorted_dictionary[50:250]:
        word = x[0]  # first element of tuple is the word
        popular_words.append(word)

    #return the 50-250 most popular words
    return popular_words

print(f"top 200 positive review words {word_counts(positive_reviews)}\n" )
print(f"top 200 negative review words {word_counts(negative_reviews)}\n")

positive_keywords = word_counts(positive_reviews)
negative_keywords = word_counts(negative_reviews)


def binary_classifier(sentence):

    # make all the words in the sentence lower case and separated 
    # classifier is only using a lowercase word bank
    sentence = sentence.lower().split()

    positive_word_count = 0
    negative_word_count = 0
    type = ''

    for word in sentence:
        if word in positive_keywords:
            positive_word_count += 1
        if word in negative_keywords:
            negative_word_count += 1

    if positive_word_count > negative_word_count:
        type = 'positive'
    elif positive_word_count < negative_word_count:
        type = 'negative'
    else: 
        type = 'undetermined'

    return type

def accuracy_of_classifier(review_file, type_of_file):
    sum_correct = 0
    for sentence in review_file:
        if binary_classifier(sentence) == type_of_file:
            sum_correct += 1

    accuracy = sum_correct / len(review_file)

    return accuracy

print(f"The accuracy of the classifier on the positive movie reviews: {accuracy_of_classifier(positive_reviews, 'positive')}")
print(f"The accuracy of the classifier on the negative movie reviews: {accuracy_of_classifier(negative_reviews, 'negative')}")








