import sys
import json
import plotly
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
import pickle

nltk.download(['stopwords', 'punkt', 'wordnet'])

app = Flask(__name__)

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "run"
        return super().find_class(module, name)

def tokenize(text):

    '''
    This function normalize, removes punctuation, tokenize, remove stopwords, lemmatize, stem, and lemmatize words in text
    
    param text: text
    ''' 
    #tokenize
  
    words = word_tokenize(text)
    
    english_stp_words = stopwords.words("english")
    lemmeds = []
    for w in words:
        w=re.sub(r"[^a-zA-Z0-9]", " ", w.lower())
        if w not in english_stp_words: #remove stopwords
            lemmed = WordNetLemmatizer().lemmatize(w.strip())#lemmatize
            lemmeds.append(lemmed)
            
    return lemmeds
    

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)

# load model
#model = joblib.load("../models/classifier.pkl")
with open('../models/classifier.pkl', 'rb') as f:
    #model = pickle.load(f)
    unpickler = MyCustomUnpickler(f)
    model = unpickler.load()

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)  
    
    request_counts = df.groupby('request').count()['message']
    request_names = list(request_counts.index)  
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=request_names,
                    y=request_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Request Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Lables"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print(classification_results)
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():

    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()