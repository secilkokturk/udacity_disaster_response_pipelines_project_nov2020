#This script includes:
#A data cleaning pipeline that:
#Loads the messages and categories datasets
#Merges the two datasets
#Cleans the data
#Stores it in a SQLite database in the specified database file path
#RUN Command: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    '''
    This function loads messages and categories datasets from csv\n
    
    :param messages_filepath: csv file name for messages
    :param categories_filepath: csv file name for categories
    '''

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
              
    # load messages dataset  
    messages = load_csv_file(messages_filepath)
    
    # load categories dataset
    categories = load_csv_file(categories_filepath)
    
    # merge datasets
    df=merge_dfs(messages,categories,'id','left')
    
    return df
    
def load_csv_file(filename):

    '''
    This function loads data from the csv file name input given\n
    ...and outputs the data df\n

    :param filename: csv file name
    '''

    print('Loading: {}!\n'.format(filename))
    df=pd.read_csv(filename)
    return df
    
def merge_dfs(df1,df2,key,howto):

    '''
    This function merges two data frames on key and howto\n
    ...and outputs the merged data df\n

    :param df1: first data frame to be merged\n
    :param df2: second data frame to be merged\n
    :param key: merge key\n
    :param howto: merge type\n
    '''

    print('Merging: datasets!\n')
    df = df1.merge(df2,on=[key],how=howto)
    return df    

def clean_data(df):

    '''
    This function cleans the disaster response dataset\n
    ...and outputs the cleaned data df\n

    :param df: data frame to be cleaned\n
    '''

    print('Cleaning the dataset!\n')

    #fix the categories
    print('Fixing the categories!\n')
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';' , expand=True)
    categories_df = pd.DataFrame(categories)
    # select the first row of the categories dataframe
    first_row = categories.iloc[0]

    # use this row to extract a list of new column names for categories
    category_column_names = first_row.str.split('-').apply(lambda x:x[0])

    # rename the columns of `categories`
    categories.columns = category_column_names

    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').apply(lambda x:x[1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    #Remove duplicates
    print('Removing the duplicates!\n')
    df = df.drop_duplicates()

    #Fill null values
    print('Filling null values!\n')
    df = df.fillna(0)

    print('Dataset is clean!\n')
    return df

      
def write_to_sqlite_db(df, sqlite_db, sqlite_table_name):

    '''
    This function writes the disaster response dataset into an sqlite db\n

    :param df: data frame to be written\n
    :param sqlite_db: db name\n
    :param sqlite_table_name: db table name\n
    '''

    print('Writing the dataset {} to sqlite db: {}\n'.format(sqlite_db, sqlite_table_name))

    df.to_sql(sqlite_db, create_engine('sqlite:///'+sqlite_table_name), index=False)

    print('Writing the dataset to sqlite db is done!\n') 
    
def main():

    if len(sys.argv) == 4:
        
        # take the file paths of the two datasets and database
        disaster_messages_fname, disaster_categories_fname, sqlite_database_name= sys.argv[1:]

        print('This script will load the data from Messages: {} and Categories: {} files and load the clean data into the sqlite database {}'.format(disaster_messages_fname, disaster_categories_fname, sqlite_database_name))
 
        df=load_data(disaster_messages_fname, disaster_categories_fname)
        
        #clean the dataset
        df= clean_data(df)
        
        #save the dataset into sqlite db
        write_to_sqlite_db(df, 'Message', sqlite_database_name)

    
    else:
        print('This script will load the data from Messages and Categories files and load the clean data into the sqlite database!\n\
        Please provide the names of the csv files Messages, Categories, and the database name as an argument.\n\
        such as "python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db"')
        
if __name__ == '__main__':
    main()