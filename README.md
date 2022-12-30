# Plagiarism Checker ( Web Engineering - Dr. Khaled Bahnasy )

  ## Steps To Run The Project

    1- This link has a video explaining how to install Flask Framework on Windows 10 " https://www.youtube.com/watch?v=uxZuFm5tmhM "

    2- In your IDE Terminal you'll need to import all libraries defined in the first lines in app.python EX: pip install sklearn. Libraries: 
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

    3- Make Sure That Python is Installed Correctly and Flask, To Run Just Write in The Terminal "flask run".

- To Terminate Ctrl+C

---------------------------------------------------------------------------------------------------------------------------------------------------------


# Plagiarism Detector
This program is a web application that allows users to check for plagiarism in texts. It compares the similarity of two PDF Files and returns a score indicating the degree of similarity.

# Features
The program uses three different methods to measure similarity: Cosine similarity, Jaccard distance, and Levenshtein distance.
The program allows users to upload PDF file and Folder Path.
The program can handle texts in English and removes stop words before comparison.

# Dependencies
This program requires the following libraries:
  - Flask: a web framework for Python
  - re: a library for working with regular expressions
  - collections: a library for working with data structures
  - math: a library for mathematical operations
  - nltk: a library for natural language processing tasks
  - sklearn: a library for machine learning tasks
  - Levenshtein: a library for calculating Levenshtein distance
  - PyPDF2: a library for reading and manipulating PDF files


# Usage
## To use this program :
  1. Run the following command: **flask run**
  2. Then, open a web browser and go to http://localhost:5000/. 
  3. You can then upload PDF File and input a Folder Path (That Contains Other PDF Files) you want to compare.
  4. Press Compare, You'll be redirected to comparison page.
  
  *- You'll Find A Folder In The Project Called /Data I've Put Some Samples For Test*
 
 # ScreenShots 
 
 ![Screenshot from 2022-12-30 15-18-24](https://user-images.githubusercontent.com/76593230/210074357-7e9391f4-7f2e-4970-963c-4a4b86e61f1d.png)

 ![Screenshot from 2022-12-30 15-17-32](https://user-images.githubusercontent.com/76593230/210074411-a7026328-6fa2-48fd-be09-f8fb8e2e5727.png)
 
 ![Screenshot from 2022-12-30 15-18-11](https://user-images.githubusercontent.com/76593230/210074399-bfd6e185-3ff6-40b5-aa03-96fe940396db.png)



  
