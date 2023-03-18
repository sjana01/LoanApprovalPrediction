## End to End Machine Learning Project

This project is simply for educational purposes. We built an application for predicting loan approval probabilty based on various features such as Gender, Marital status, Income, Loan amount, Loan term, Credit history, etc.

We use the pipeline object in scikit-learn to create a data transformation pipeline. This transforms the raw data into trainable data with the help of various methods like Imputer, Encoder, Scaler, etc. We save the data preprocessing pipeline in a pickle file for processing new data in future.

We use multiple classification algorithms for training and find the best model. We will also implement some hyper-parameter tuning using Grid Search. We save the best model and best parameters in a pickle file for future use.

We then build an Flask app and deploy the application on AWS Beanstalk.