# Disaster Response Pipeline Project

### This project is developed in order to fulfill the Udacity Data Science Nanodegree:

The disaster messages and categories data from Figure Eight is analyzed and used for training a model that would classify messages.

The machine learning pipeline runs a Random Forest on a grid, finds the best parameters and outputs the model to a pickle file, which is used by a flask web app.

### Project Files:

1) ETL Pipeline:
The process_data.py module reads, merges the messages and categories datasets. Cleans the text data and splits the categories, and then puts it in an sqlite db called DisasterReponse.db.

1) ML Pipeline:
The train_classifier.py loads the data coming from the ETL pipeline, builds a pipeline to find the best parameters for 80% of the data available using a random forest classifier, outputs the accuracies for each category for the test set, and saves the model as a pickle file classifier.pkl.

1) Flask Web App:
The web app uses the pkl file, shows two visualisations on the front page, and classifies messages according to the model. The output shows multiple fit categories lighted in green for each message, such as: 
Message: "Is wind power’s future in deep water?"
Categories: Related, Weather Related

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py` and go to http://0.0.0.0:3001/
	
   In windows, run 'set run FLASK_APP=run.py', then run 'flask run' to see the web app on http://127.0.0.1:5000/ ('flask run -h 0.0.00 -p 3001' should also work but did not run for me for an unknown reason)
