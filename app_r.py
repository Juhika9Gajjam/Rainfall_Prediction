import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model_cls = pickle.load(open('model_RF.pkl', 'rb'))
model_reg = pickle.load(open('model_xgb_3sets.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    #For rendering results on HTML GUI

    int_features = [x for x in request.form.values()]
    #final_features = [np.array(int_features)]
    ss=StandardScaler()
    #final_features=ss.transform(final_features)
    prediction = model_cls.predict(int_features)

    #output = round(prediction[0], 2)
    #flag= prediction[0]
    flag= prediction[0]
    if flag:
        return render_template('index.html', prediction_text='There is high chance to Rain on this Day')
    else:
        return render_template('index.html', prediction_text='There is no Rainfall today')

if __name__ == "__main__":
    app.run(debug=True)