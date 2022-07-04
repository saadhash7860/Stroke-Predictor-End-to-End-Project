import pandas as pd
import pickle
import numpy as np
import sklearn
from flask import Flask,render_template , request,jsonify

app = Flask(__name__)

@app.route("/")

def Home():
    return render_template('index.html')

@app.route('/predict' ,methods = ['POST','GET'])
def results():
    gender = request.form['gender']
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = request.form['ever_married']
    work_type  = request.form['work_type']
    Residence_type = request.form['Residence_type']
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = request.form['smoking_status']

    # gender
    if "Male" == gender:
        gender_Male = 1
        gender_Other = 0


    elif gender == "Other":
        gender_Male = 0
        gender_Other = 1
    else:
        gender_Male=0
        gender_Other=0

    # married
    if ever_married=="Yes":
        ever_married_Yes = 1
    else:
        ever_married_Yes= 0

    # work_type
    if work_type=='Self-employed':
        work_type_Never_worked= 0
        work_type_Private= 0
        work_type_Self_employed= 1
        work_type_children=0
    elif work_type == 'Private':
        work_type_Never_worked = 0
        work_type_Private = 1
        work_type_Self_employed = 0
        work_type_children=0
    elif work_type =="children":
         work_type_Never_worked = 0
         work_type_Private = 0
         work_type_Self_employed = 0
         work_type_children=1
    elif work_type=="Never_worked":
        work_type_Never_worked = 1
        work_type_Private = 0
        work_type_Self_employed = 0
        work_type_children=0
    else:
        work_type_Never_worked = 0
        work_type_Private = 0
        work_type_Self_employed = 0
        work_type_children=0
     # residence type
    if Residence_type =="Urban":
        Residence_type_Urban=1
    else:
        Residence_type_Urban=0
     # smoking status
    if smoking_status == 'formerly smoked':
        smoking_status_formerly_smoked = 1
        smoking_status_never_smoked = 0
        smoking_status_smokes = 0
    elif smoking_status == 'smokes':
        smoking_status_formerly_smoked = 0
        smoking_status_never_smoked = 0
        smoking_status_smokes = 1
    elif smoking_status  == "never smoked":
        smoking_status_formerly_smoked = 0
        smoking_status_never_smoked = 1
        smoking_status_smokes = 0
    else:
        smoking_status_formerly_smoked = 0
        smoking_status_never_smoked = 0
        smoking_status_smokes = 0

    X = np.array([[age, hypertension, heart_disease	,avg_glucose_level,bmi ,gender_Male , gender_Other, ever_married_Yes,work_type_Never_worked,work_type_Private,work_type_Self_employed,work_type_children,Residence_type_Urban, smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes]])


    sc = pickle.load(open('sc.pkl','rb'))

    X_std = sc.transform(X)

    
    model = pickle.load(open('model_stroke_predictor.pkl','rb'))
    
    Y_prediction = model.predict(X_std)
    if Y_prediction==0:
        Y_prediction = "NO Stroke" 
    else:
        Y_prediction = "Stroke"

    return jsonify({'Model Prediction': (Y_prediction)})

    
if __name__ == "__main__":
        app.run(debug=True)




