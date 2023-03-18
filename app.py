import os, sys
from flask import Flask, request, render_template

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline



application = Flask(__name__)

app = application

# Route for a homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictddata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Gender = request.form.get('Gender'),
            Married = request.form.get('Married'),
            Dependents = request.form.get('Dependents'), 
            Education = request.form.get('Education'), 
            Self_Employed = request.form.get('Self_Employed'), 
            ApplicantIncome = request.form.get('ApplicantIncome'),  
            CoapplicantIncome = request.form.get('CoapplicantIncome'), 
            LoanAmount = request.form.get('LoanAmount'),  
            Loan_Amount_Term = request.form.get('Loan_Amount_Term'), 
            Credit_History = request.form.get('Credit_History'), 
            Property_Area = request.form.get('Property_Area')
        )
        pred_df = data.get_data_as_df()
        # print(pred_df)

        pr = PredictPipeline()
        results = pr.predict(pred_df)
        decision = ''
        if results[0] == 0:
            decision = 'Not Approved'
        else:
            decision = 'Approved'
        print(results)
        return render_template('home.html', decision = decision)
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)

