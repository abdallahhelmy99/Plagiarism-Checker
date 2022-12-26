from flask import Flask, render_template, request
import os
import re
from collections import Counter
from math import sqrt
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn.metrics import jaccard_score
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as edit_distance




app = Flask(__name__)
app.static_folder = 'static'



def get_cosine_similarity(text1, text2):
    # create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # fit the vectorizer on the text data
    X = vectorizer.fit_transform([text1, text2])

    # compute the cosine similarity
    similarity = cosine_similarity(X[0], X[1])[0][0]
    
    return similarity




def get_jaccard_distance(text1, text2):
    # split the text into sentences
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)

    # convert the sentences to sets
    s1 = set(sentences1)
    s2 = set(sentences2)

    # compute the Jaccard distance
    distance = jaccard_score(s1, s2)
    
    return distance


def get_edit_distance(text1, text2):
    # split the text into sentences
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)

    # compute the average edit distance of the sentences
    distance = sum(edit_distance(s1, s2) for s1, s2 in zip(sentences1, sentences2)) / len(sentences1)
    
    return distance

# create a set of stop words
stop_words = set(stopwords.words('english'))

# def get_common_words(text1, text2):
#     # split the text into words
#     words1 = text1.split()
#     words2 = text2.split()
    
#     # remove stop words
#     filtered_words1 = [word for word in words1 if word.lower() not in stop_words]
#     filtered_words2 = [word for word in words2 if word.lower() not in stop_words]
    
#     # count the frequency of each word
#     frequency1 = Counter(filtered_words1)
#     frequency2 = Counter(filtered_words2)
    
#     # find the common words
#     common = set(frequency1.keys()) & set(frequency2.keys())
#     if not common:
#         return []
    
#     # return the common words
#     common_words = [word for word in common]
#     return common_words


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():

    file = request.files['file_path']
    folder_path = request.form['folder_path']

    text1 = file.read()
    text1 = text1.decode()
    text1 = re.sub(r'[^\w\s]', '', text1)
    
    files = os.listdir(folder_path)
    
    results = []
    for file in files:
        with open(f'{folder_path}/{file}', 'r') as f:
            text2 = f.read()
        text2 = re.sub(r'[^\w\s]', '', text2)
        #common_words = get_common_words(text1, text2)
        similarity = get_jaccard_distance(text1, text2) 
        results.append({ 'file': file, 'similarity': similarity })

    return render_template('result.html', results=results)