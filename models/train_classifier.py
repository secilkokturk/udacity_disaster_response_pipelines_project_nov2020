#This script includes:
#A machine learning pipeline that:
#Loads data from the SQLite database
#Splits the dataset into training and test sets
#Builds a text processing and machine learning pipeline
#Trains and tunes a model using GridSearchCV
#Outputs results on the test set
#Exports the final model as a pickle file
#RUN Command: python train_classifier.py ../data/DisasterResponse.db classifier.pkl

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from sklearn.metrics import classification_report

nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])

class StartingVerbExtract(BaseEstimator, TransformerMixin):
    '''
    Finds the starting verb in text
    '''
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


class TextLengthExtract(BaseEstimator, TransformerMixin):
    '''
    Finds the text length of each df cell
    '''
    def fit(self, X, y=None):
    	'''
    	fit self
    	'''
    	return self

    def transform(self, X):
    	'''
    	Finds the text length of each cell in df
    	'''
    	X_length = pd.Series(X).str.len()
    	return pd.DataFrame(X_length)


def load_data(database_filepath):

    '''
    This function loads data from the database file name input given\n
    ...and outputs the data: X (feature columns), Y (target column), category names\n

    :param database_filepath: database file name
    '''
    # load data from database file path
    
    #check table name
    # from sqlalchemy import inspect
    # inspector = inspect(create_engine('sqlite:///'+database_filepath))
    # schemas = inspector.get_schema_names()

    # for schema in schemas:
        # print("schema: %s" % schema)
        # for table_name in inspector.get_table_names(schema=schema):
            # print("Table: %s" % table_name)
            # for column in inspector.get_columns(table_name, schema=schema):
                # print("Column: %s" % column)


    df = pd.read_sql_table('Message',create_engine('sqlite:///'+database_filepath))
    
    #print('Columns:\n')
    #print(df.columns)
    
    #print('Investigate genre column:\n')
    #df.groupby(['genre']).size()
    
    print('Encode genre column:\n')
    one_hot = pd.get_dummies(df['genre'])
    df = df.join(one_hot)
    df = df.drop('genre',axis = 1)
    print('Genre column encoded!\n')
    print('Columns:\n')
    #print(df.columns)

    X = df['message']
    Y = df.drop(columns=['id','message','original'])
    
    return X, Y, Y.columns


def tokenize(text):
    '''
    This function normalize, removes punctuation, tokenize, remove stopwords, lemmatize, stem, and lemmatize words in text
    
    param text: text
    ''' 
    # use a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text: vectorize and then apply TF-IDF to the text

    #case normalize
    text = text.lower() 

    #remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    #tokenize
    words = word_tokenize(text)
    
    
    lemmeds = []
    for w in words:
        if w not in stopwords.words("english"): #remove stopwords
            #stemmed = PorterStemmer().stem(w) #stem
            lemmed = WordNetLemmatizer().lemmatize(w)#lemmatize
            lemmeds.append(lemmed)
            
    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmeds]

    clean_lems = []
    for lem in lemmed:
        clean_lems.append(lem)
    return clean_lems
    
def build_model():
    '''
    This function creates a classifier pipeline
    ''' 

    #build a pipeline that processes text and then performs multi-output classification on the 39 categories in the dataset:
    #related, request, offer, aid_related, medical_help, medical_products, search_and_rescue, security, military, child_alone, water, food, shelter, clothing, money, missing_people, refugees, death, other_aid, infrastructure_related, transport, buildings, electricity, tools, hospitals, shops, aid_centers, other_infrastructure, weather_related, floods, storm, fire, earthquake, cold, other_weather, direct_report,
    # (encoded from genre) direct, news, social
    #The script uses a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text. This function is used in the machine learning pipeline to vectorize and then apply TF-IDF to the text.

    #create the pipeline for text transformation
    pipeline = Pipeline([
        ('features', FeatureUnion
            ([

                ('text_pipeline', 
                
                Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                        ])
                )
            ])
            ,
            ('starting_verb', StartingVerbExtract()),
            ('text_len', TextLengthExtract())
        ),

        ('m_clf', MultiOutputClassifier(RandomForestClassifier(verbose=1)))
    ])
    #optimization & evaluation: GridSearchCV is used to find the best parameters for the model

    # Set up the search grid
    parameters = {
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [100, 150, 200],
        'clf__min_samples_split': [2, 3, 4],
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000)
    }
    # Initialize GridSearch cross validation object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):

    '''
    Evaluate the performance of the model performance for all category target columns
    
    :param model: model
    :param X_test: test set
    :param Y_test: target set
    :param category_names: target categories
    ''' 
    # Predict the test set
    Y_pred = model.predict(X_test)

    Y_pred = pd.DataFrame(Y_pred,columns=category_names)
    
    #output: f1 score, precision and recall for each category
    # The TF-IDF pipeline is only trained with the training data. The f1 score, precision and recall for the test set is outputted for each category. 
    for col in category_names:
        print(f'Performance of Column:{col}\n')
        print(classification_report(Y_test[col],Y_pred[col]))


def save_model(model, model_filepath):
    # store the classifier into a pickle file to the specified model file path
    '''
    This function saves model to a pickle file
    
    :param model: model
    :param model_filepath: pickle file name
    '''
    pickle.dump(model, open(model_filepath, 'wb'))  
    

def main():


    if len(sys.argv) == 3:
    
        # take the database file path and model file path
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        #check the tokenize function
        for m in X[:20]:
            print(m)
            print(tokenize(m))
            
            
        #Split the dataset into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # create and train a classifier      
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # store the classifier into a pickle file to the specified model file path
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()