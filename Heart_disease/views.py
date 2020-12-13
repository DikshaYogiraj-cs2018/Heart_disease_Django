from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def home(request):
    return render(request,'home.html')

def predict(request):
    return render(request,'predict.html')

def result(request):
        df = pd.read_csv('static/data.csv')
        data = df.values
        X = data[:, :-1]
        Y = data[:, -1:]

        value = ''

        if request.method == 'POST':

            age = request.POST['age']
            sex =  request.POST['sex']
            cp = request.POST['cp']
            trestbps = request.POST['trestbps']
            chol = request.POST['chol']
            fbs = request.POST['fbs']
            restecg = request.POST['restecg']
            thalach = request.POST['thalach']
            exang = request.POST['exang']
            oldpeak = request.POST['oldpeak']
            slope = request.POST['slope']
            ca = request.POST['ca']
            thal = request.POST['thal']

            user_data = np.array(
                (age,
                 sex,
                 cp,
                 trestbps,
                 chol,
                 fbs,
                 restecg,
                 thalach,
                 exang,
                 oldpeak,
                 slope,
                 ca,
                 thal)
            ).reshape(1, 13)

            rf = RandomForestClassifier(
                n_estimators=16,
                criterion='entropy',
                max_depth=9
            )

            rf.fit(np.nan_to_num(X), np.ravel(Y))
            predictions = rf.predict(user_data)
            print(predictions)

            if int(predictions[0]) == 1:
                value = 'Have Heart Disease'
            elif int(predictions[0]) == 0:
                value = "Not likely to have Heart Disease"

        return render(request,
                      'predict.html',{'prediction':value})