In this project i used the Emotion_dataset to train my model.
i cleaned the data set using NLTK and Swifter to remove stopwords,Url and etc to have a better text to train my model. 
and then i splitted the data to train and test parts and used the models like LogisticRegression,LinearSVC,MultinomialNB and SGDClassifier.
i used the metrics to have a comparison between the models and after considering the results i use svm as my main model for training.
then i saved the model using joblib and with streamlit now it has a little UI to work with.
