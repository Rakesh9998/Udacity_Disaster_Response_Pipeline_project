# Disaster Response Pipeline Project


### Introduction

#### 1 .ETL
The first part of the project is the Extract, Transform, and Load process. Here, I have read the dataset, cleaned the data, and then store it in a SQLite database. Data cleaning is done with pandas. Then loaded the data into an SQLite database.

Final code is there in **/data/process_data.py.**

#### 2.Machine Learning Pipeline
In this part, I have split the data into a training set and a test set. Then, I have created a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model. Finally, I have exported my model to a pickle file. 

Final machine learning code is there in **/models/train_classifier.py.**

#### 3.Flask App
This is the last step where I have displayed my results in a Flask web app.
The codes are contained in the **app** folder.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files Used
1. process_data.py - ETL is done 

2. train_classifier.py - Machine learning pipeline is done

3. run.py - code to run web app

4. disaster_messages.csv,disaster_categories.csv - the csv files containing the messages and the categories these messages belong to

5. DisasterResponse.db - sql database after data preprocessing

6. classifier.pkl - pickle file in which my machine learning model is exported.
