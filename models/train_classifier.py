import warnings
warnings.filterwarnings("ignore")
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,hamming_loss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
from sklearn import metrics
from sklearn import grid_search
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
nltk.download('stopwords')
nltk.download('punkt')
import pickle


def load_data(database_filepath):
    
    '''This function loads the data from the sql database and also creates X and Y
    for machine learning models. X is input to the model and Y should be the output of the model.'''
    
    # load data from database
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table('Cleaned_dataset3', engine)

    # Create X and Y
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'], axis=1).values
    category_names = list(df.drop(['id','message','original','genre'], axis=1))
    return X, Y, category_names


def tokenize(text):
    
    '''This function takes the text as input and returns it after tokenizing
    Credit: Udacity Nanodegree Case Studies'''
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    '''This function is used to build machine learning model for multilabel classes'''
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=0.00009, smooth_idf=True, norm="l2",tokenizer =tokenize, sublinear_tf=False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = {
        'tfidf__ngram_range': ((1, 1),(1,2)),
        'clf__estimator__n_estimators': [10,50,],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters , verbose = 2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''This function evaluates the model based on hamming loss,accuracy score , f1 score,
    macro average quality numbers , micro average quality numbers ,precision and recall values'''
    
    y_cap = model.predict(X_test)
    precision = precision_score(Y_test, y_cap, average='micro')
    recall = recall_score(Y_test, y_cap, average='micro')
    f1 = f1_score(Y_test, y_cap, average='micro')
    
    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    
    precision = precision_score(Y_test, y_cap, average='macro')
    recall = recall_score(Y_test, y_cap, average='macro')
    f1 = f1_score(Y_test, y_cap, average='macro')
    
    print("Macro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    
    #print("Accuracy :",accuracy_score(Y_test, y_cap))
    #print("Hamming loss ",hamming_loss(Y_test,y_cap))
    print(metrics.classification_report(y_cap, Y_test, target_names=category_names))
    #Displaying accuracy score and hamming loss of all the features
    for i in range(36):
        current_category = category_names[i]
        acc_score = accuracy_score(Y_test[:,i],y_cap[:,i])
        ham_loss = hamming_loss(Y_test[:,i],y_cap[:,i])
        print("Accuracy Score of {0} is {1} and Hamming loss of {0} is {2}".format(current_category,acc_score,ham_loss))


def save_model(model, model_filepath):
    
    '''This function saves the model to a pickle file'''
    
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
