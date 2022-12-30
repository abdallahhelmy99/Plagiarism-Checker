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
from nltk.tokenize import word_tokenize




app = Flask(__name__)
app.static_folder = 'static'


stop_words = set(stopwords.words('english'))
#define a function to remove stop words
def remove_stop_words(text):
    # split the text into words
    words = text.split()
    
    # remove stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # join the words
    text = ' '.join(filtered_words)
    
    return text

def get_cosine_similarity(text1, text2):
    # Split the sentences into words
    words1 = text1.split()
    words2 = text2.split()
    
    # Calculate the frequencies of the words in each sentence
    frequency1 = Counter(words1)
    frequency2 = Counter(words2)
    
    # Find the common words between the two sentences
    common = set(frequency1.keys()) & set(frequency2.keys())
    
    # If there are no common words, return 0
    if not common:
        return 0
    
    # Calculate the dot product and magnitudes of the frequency vectors
    dot_product = sum(frequency1[word] * frequency2[word] for word in common)
    magnitude1 = sqrt(sum(frequency1[word]**2 for word in frequency1.keys()))
    magnitude2 = sqrt(sum(frequency2[word]**2 for word in frequency2.keys()))
    
    # Calculate the cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    
    return similarity


# def get_cosine_similarity(text1, text2):
#     words1 = word_tokenize(text1)
#     words2 = word_tokenize(text2)
    
#     frequency1 = Counter(words1)
#     frequency2 = Counter(words2)
    
#     common = set(frequency1.keys()) & set(frequency2.keys())
    
#     if not common:
#         return 0
    
#     dot_product = sum(frequency1[word] * frequency2[word] for word in common)
    
#     magnitude1 = sqrt(sum(frequency1[word]**2 for word in frequency1.keys()))
#     magnitude2 = sqrt(sum(frequency2[word]**2 for word in frequency2.keys()))
    
#     similarity = dot_product / (magnitude1 * magnitude2)
    
#     return similarity

# create a function calculates cosine similarity
# def get_cosine_similarity(text1, text2):
#     words1 = text1.split()
#     words2 = text2.split()
    
#     frequency1 = Counter(words1)
#     frequency2 = Counter(words2)
    
#     common = set(frequency1.keys()) & set(frequency2.keys())
    
#     if not common:
#         return 0
    
#     dot_product = sum(frequency1[word] * frequency2[word] for word in common)
    
#     magnitude1 = sqrt(sum(frequency1[word]**2 for word in frequency1.keys()))
#     magnitude2 = sqrt(sum(frequency2[word]**2 for word in frequency2.keys()))
    
#     similarity = dot_product / (magnitude1 * magnitude2)
    
#     return similarity


def get_jaccard_distance(text1, text2):
    # Split the texts into words
    words1 = text1.split()
    words2 = text2.split()
    
    # Find the intersection and union of the words in the texts
    intersection = set(words1) & set(words2)
    union = set(words1) | set(words2)
    
    # Calculate the Jaccard distance
    jaccard_score = 1 - len(intersection) / len(union)
    
    return jaccard_score



def get_edit_distance(text1, text2):
    # split the text into sentences
    sentences1 = sent_tokenize(text1)
    sentences2 = sent_tokenize(text2)

    # compute the average edit distance of the sentences
    distance = sum(edit_distance(s1, s2) for s1, s2 in zip(sentences1, sentences2)) / len(sentences1)
    
    return distance

# create a set of stop words



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

import PyPDF2

def get_pdf_text(pdf_file):
    # open the PDF file
    pdf = PyPDF2.PdfReader(pdf_file)

    # get the number of pages
    num_pages = len(pdf.pages)

    # initialize an empty string to store the text
    text = ""

    # iterate through the pages
    for i in range(num_pages):
        # get the text of the page
        page = pdf.pages[i]
        page_text = page.extract_text()

        # add the page text to the overall text
        text += page_text

    # return the text
    return text



def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        # Create a PDF object
        pdf = PyPDF2.PdfReader(f)
        # Extract the text from all the pages
        text = '\n'.join([pdf.getPage(page_num).extract_text() for page_num in range(pdf.getNumPages())])
        return text

def extract_text_from_pdfs_in_folder(folder_path):
    # Get a list of all the PDF files in the folder
    pdf_filenames = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    # Extract the text from each PDF file and store it in a dictionary
    pdf_texts = {}
    for pdf_filename in pdf_filenames:
        pdf_path = os.path.join(folder_path, pdf_filename)
        text = extract_text_from_pdf(pdf_path)
        pdf_texts[pdf_filename] = text
    return pdf_texts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])


def compare():
    file = request.files['file_path']
    folder_path = request.form['folder_path']

    text1 = get_pdf_text(file)

    files = os.listdir(folder_path)
    
    results = []
    for file in files:
        #extract_text_from_pdfs_in_folder(file)
        try:
            with open(f'{folder_path}/{file}', 'r', encoding='utf-8') as f:
                pdf2 = PyPDF2.PdfReader(f'{folder_path}/{file}')
                text2 = ""
                for page in range(len(pdf2.pages)):
                    text2 += pdf2.pages[page].extract_text()
                #text2 = re.sub(r'[^\w\s]', '', text2)
        except UnicodeDecodeError:
            with open(f'{folder_path}/{file}', 'r', encoding='latin-1') as f:
                pdf2 = PyPDF2.PdfReader(f'{folder_path}/{file}')
                text2 = ""
                for page in range(len(pdf2.pages)):
                    text2 += pdf2.pages[page].extract_text()
                #text2 = re.sub(r'[^\w\s]', '', text2)
        
        
        text1_new = remove_stop_words(text1)
        text2_new = remove_stop_words(text2)
        common_words = get_common_words(text1_new, text2_new)
        common_sentences = get_common_sentences(text1, text2)
        cosineSimilarity = get_cosine_similarity(text1_new, text2_new)
        print(text1_new)
        print(text2_new)
        jaccard_score = get_jaccard_distance(text1_new, text2_new)
        editDistance = get_edit_distance(text1_new, text2_new)
        
        results.append({ 'file': file, 'jaccard': jaccard_score, 'editDistance': editDistance, 'cosineSimilarity': cosineSimilarity, 'common_sentences': common_sentences, 'common_words': common_words })

    return render_template('result.html', results=results)
