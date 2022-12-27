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
import PyPDF2




app = Flask(__name__)
app.static_folder = 'static'



# create a function calculates cosine similarity
def get_cosine_similarity(text1, text2):
    words1 = text1.split()
    words2 = text2.split()
    
    frequency1 = Counter(words1)
    frequency2 = Counter(words2)
    
    common = set(frequency1.keys()) & set(frequency2.keys())
    
    if not common:
        return 0
    
    dot_product = sum(frequency1[word] * frequency2[word] for word in common)
    
    magnitude1 = sqrt(sum(frequency1[word]**2 for word in frequency1.keys()))
    magnitude2 = sqrt(sum(frequency2[word]**2 for word in frequency2.keys()))
    
    similarity = dot_product / (magnitude1 * magnitude2)
    
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


# create a function to find the common sentences
def get_common_sentences(text1, text2):
    # split the text into sentences
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)
    
    # remove stop words
    filtered_sentences1 = [sentence for sentence in sentences1 if sentence.lower() not in stop_words]
    filtered_sentences2 = [sentence for sentence in sentences2 if sentence.lower() not in stop_words]
    
    # count the frequency of each sentence
    frequency1 = Counter(filtered_sentences1)
    frequency2 = Counter(filtered_sentences2)
    
    # find the common sentences
    common = set(frequency1.keys()) & set(frequency2.keys())
    if not common:
        return []
    
    # return the common sentences
    common_sentences = [sentence for sentence in common]
    return common_sentences

def get_common_words(text1, text2):
    # split the text into words
    words1 = text1.split()
    words2 = text2.split()
    
    # remove stop words
    filtered_words1 = [word for word in words1 if word.lower() not in stop_words]
    filtered_words2 = [word for word in words2 if word.lower() not in stop_words]
    
    # count the frequency of each word
    frequency1 = Counter(filtered_words1)
    frequency2 = Counter(filtered_words2)
    
    # find the common words
    common = set(frequency1.keys()) & set(frequency2.keys())
    if not common:
        return []
    
    # return the common words
    common_words = [word for word in common]
    return common_words


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])


def compare():
    file = request.files['file_path']
    folder_path = request.form['folder_path']

    # Create a PDF object from the uploaded file
    pdf = PyPDF2.PdfReader(file)
    # Extract the text from the PDF
    text1 = ""
    for page in range(len(pdf.pages)):
        text1 += pdf.pages[page].extract_text()
        #text1 += pdf.getPage(page).extractText()
    # Remove punctuation from the text
    text1 = re.sub(r'[^\w\s]', '', text1)

    files = os.listdir(folder_path)
    
    results = []
    for file in files:
       try:
    with open(f'{folder_path}/{file}', 'r', encoding='utf-8') as f:
        text2 = f.read()
except UnicodeDecodeError:
    with open(f'{folder_path}/{file}', 'r', encoding='latin-1') as f:
        text2 = f.read()

        common_words = get_common_words(text1, text2)
        common_sentences = get_common_sentences(text1, text2)
        cosineSimilarity = get_cosine_similarity(text1, text2)
        if len(text1) == len(text2):
            jaccard = get_jaccard_distance(text1, text2)
        else:
            jaccard = 0
        editDistance = get_edit_distance(text1, text2)
        results.append({ 'file': file, 'jaccard': jaccard, 'editDistance': editDistance, 'cosineSimilarity': cosineSimilarity, 'common_sentences': common_sentences, 'common_words': common_words })

    return render_template('result.html', results=results)
