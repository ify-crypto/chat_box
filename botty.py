# import nltk
import random
import streamlit as st
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()



# Download required NLTk data
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt_tab')

samsung= pd.read_csv('Samsung Dialog.txt',sep = ':',header = None)

cust= samsung.loc[samsung[0] == 'Customer']
sales= samsung.loc[samsung[0] == 'Sales Agent']

sales = sales [1].reset_index(drop=True)
cust = cust[1].reset_index(drop=True)

new_data = pd.DataFrame()
new_data['Questions'] = cust
new_data['Answers'] = sales

# Define a function for text preprocessing( includig lemmatization)
def preprocess_text(text):
    # identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)

    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower())for word in nltk.word_tokenize(sentence)if word.isalnum()]
         # Turns to basic root -each word in the tokenized word found in the tokenized sentence -if they are alphanumeric
         # The code above does the following:
         # identifies every word in the sentence
         # Turns it to a lower case
         # Lemmatizes it if the word is alphanumeric




        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)

    return' '.join(preprocessed_sentences)


new_data['tokenized Questions'] = new_data['Questions'].apply(preprocess_text)


xtrain = new_data['tokenized Questions'].to_list()

tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)


bot_greeting = ['Hello User! Do you have any questions?',
                'Hey  you! tell me what you want',
                'I am like a genie in a bottle. Hit me with your question.',
                'Hi! how can i help you today?']

bot_farewell= ['Thanks for your usage... bye.',
              'I hope you had a good experience.',
              'Have a great day and keep enjoying Samsung.']


human_greeting= ['hi','hello', 'good day', 'hey', 'hola' ]

human_exit = ['thank you', 'thanks', 'bye bye', 'goodbye', 'quit']


import random
random_greeting = random.choice(bot_greeting)
random_farewell = random.choice(bot_farewell) 


st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size: 60px; font-family: Monospace'>ORGANIZATION CHAT BOX </h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>Built by IFEYINWA</h4>", unsafe_allow_html = True)


st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)


st.header('Project Background Information',divider= True)
st.write("A chatbox organization refers to a structured system or company that develops and manages chatbot-based communication platforms. These organizations leverage artificial intelligence (AI) and natural language processing (NLP) to automate customer service, enhance user engagement, and improve business efficiency.")


st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)


col1,col2 = st.columns(2)
col2.image('pngwing.com.png')

userPrompt = st.chat_input('Ask your Question')
if userPrompt:
    col1.chat_message('ai').write(userPrompt)

    userPrompt= userPrompt.lower()
    if userPrompt in human_greeting:
        col1.chat_message("human").write(random_greeting)
    elif userPrompt in human_exit:
        col1.chat_message("human").write(random_farewell)
    else:
        proUserInput = preprocess_text(userPrompt)
        vect_user = tfidf_vectorizer.transform([proUserInput])
        similarity_scores = cosine_similarity(vect_user,corpus)
        most_similar_index = np.argmax(similarity_scores)
        col1.chat_message("human").write(new_data['Answers'].iloc[most_similar_index])
