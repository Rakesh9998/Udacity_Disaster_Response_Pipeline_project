import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    
    '''This function is used to load 2 CSV files,
    merge them into two dataframe and return the 
    dataframe.'''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = categories.merge(messages, on='id')

    return df

def proper_multilabel_feature(x):
    
    '''This function takes an integer as input
    and then return the same integer if it is less than
    or equal to 1. Else it returns 1'''
    
    if x>1:
        return 1
    else:
        return x


def clean_data(df):
    
    '''This fuction takes the dataframe as input and separetes the different categories ,
    preprocess these features and convert them into proper multilabel features and 
    drop the duplicates if any'''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    category_colnames=[]

    # Extracting a list of new column names for categories.
    for i in range(36):
        category_colnames.append(row[i][:-2])
    category_colnames=pd.Series(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames

    # set each value to be the last character of the string
    for i in range(26386):
        for j in range(36):
            categories.iloc[i][j]=categories.iloc[i][j][-1:]

    # convert column from string to numeric and preprocess these features and convert them into proper multilabel features
    
    for i in range(26386):
        for j in range(36):
            categories.iloc[i][j]=proper_multilabel_feature(int(categories.iloc[i][j]))

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    categories['row_no'] = range(0, len(categories) )
    df['row_no'] = range(0, len(df) )
    result=categories.merge(df, on='row_no')

    result.drop(['row_no'], axis=1, inplace=True)

    # drop duplicates
    result = result.drop_duplicates(keep='first')

    return result


def save_data(df, database_filename):
    
    '''This function saves the cleaned dataset into sql database'''
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Cleaned_dataset1', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()