from unittest import result
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
RF = pickle.load(open('loan_model_RF.pkl', 'rb'))
KNN = pickle.load(open('loan_model_KNN.pkl','rb'))
SVC = pickle.load(open('loan_model_SVC.pkl','rb'))
LR = pickle.load(open('loan_model_LR.pkl','rb'))


@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    Gender = 1 if request.form['Gender'].lower() == 'male' else 0
    Married = 1 if request.form['Married'].lower() == 'yes' else 0
    Education = 1 if request.form['Education'].lower() == 'graduate' else 0
    Self_Employed = 1 if request.form['Self_Employed'].lower() == 'yes' else 0
    ApplicantIncome = float(request.form['ApplicantIncome'])
    CoapplicantIncome = float(request.form['CoapplicantIncome'])
    LoanAmount = float(request.form['LoanAmount'])
    Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
    Credit_History = float(request.form['Credit_History'])

    # Example property area mapping
    area = request.form['Property_Area'].lower()
    if area == 'urban': Property_Area = 2
    elif area == 'semiurban': Property_Area = 1
    else: Property_Area = 0

    Dependents = request.form['Dependents']
    Dependents = 3 if Dependents == '3+' else int(Dependents)

    # Final input (order must match training!)
    input_data = np.array([[Gender, Married, Education, Self_Employed, ApplicantIncome,
                        CoapplicantIncome, LoanAmount, Loan_Amount_Term,
                        Credit_History, Property_Area, Dependents]])

    prediction1 = RF.predict(input_data)[0]
    prediction2 = KNN.predict(input_data)[0]
    prediction3 = SVC.predict(input_data)[0]
    prediction4 = LR.predict(input_data)[0]

    # Example: Each model gives 0 (Rejected) or 1 (Approved)
    votes = [prediction1, prediction2, prediction3, prediction4]
    final_prediction = 1 if votes.count(1) > votes.count(0) else 0

    result_RF = "Loan Approved ✅" if prediction1 == 1 else "Loan Rejected ❌"
    result_KNN = "Loan Approved ✅" if prediction1 == 1 else "Loan Rejected ❌"
    result_SVC = "Loan Approved ✅" if prediction3 == 1 else "Loan Rejected ❌"
    result_LR = "Loan Approved ✅" if prediction4 == 1 else "Loan Rejected ❌"
    result_final = "Loan Approved ✅" if final_prediction == 1 else "Loan Rejected ❌"

    


    return render_template('form.html',
                            result_RF=result_RF,
                            result_SVC = result_SVC, 
                            result_KNN = result_KNN , 
                            result_LR = result_LR,
                            result_final = result_final)

if __name__ == "__main__":
    app.run(debug=True)