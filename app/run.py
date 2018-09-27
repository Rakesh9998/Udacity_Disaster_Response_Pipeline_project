import json
import plotly
import pandas as pd
from nltk.corpus import stopwords
from collections import *


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Cleaned_dataset', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Extracting the most common words from messages
    Cleaned_message =tokenize(' '.join(df['message']))
    Cleaned_message = [x for x in Cleaned_message if x not in stopwords.words("english") and x.isalpha() and x not in ["ha","wa","like","would","u"]]
    Cleaned_message = pd.Series(Cleaned_message)
    Most_common_words_frequency = Cleaned_message.value_counts()[0:10]
    Most_common_words = list(Most_common_words_frequency.index) 
    
    #Extracting the most common categorical problem encountered 
    categories = df.drop(['id','message','original','genre'], axis=1)
    for column in categories:
        categories[column] = categories[column].apply(lambda x: 1 if x==1 else None)
    
    Category_count=categories.count().sort_values(ascending=False)[0:10]
    Category_name =list(Category_count.index)
    
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
                    x=Most_common_words,
                    y=Most_common_words_frequency
                )
            ],

            'layout': {
                'title': 'Distibution of most Common Words used in Disaster Messages',
                'yaxis': {
                    'title': "Word Frequency"
                },
                'xaxis': {
                    'title': "Words in Disaster messages"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=Category_name,
                    y=Category_count
                )
            ],

            'layout': {
                'title': 'Distibution of most Common Categories of problems faced during disaster',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category_names"
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
    classification_results = dict(zip(df.columns[0:36], classification_labels))

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
