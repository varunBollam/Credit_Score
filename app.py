import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Loading model in VS code
app=Flask(__name__)
RF=pickle.load(open('RF.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predit_api',methods=['POST'])
def predit_api():
    data=request.json['data']
    print(data)
    data=np.array(list(data.values())).reshape(1,-1)
    #data=data.values([1]).astype(float)
    #D1=np.array(list(data.values())).astype(float)
    #data=np.array(list(data.values())).reshape(1,-1)
    #new_data=scalar.transform(np.array(list(data.values())[:]).reshape(1,-1))
    #newdata = [float(x) for x in data.values[0]]
    #data=json.dumps(data) 
    #data=np.array(list(data.values()))
    #data=data.reshape(1,-1)
    #data=np.array(list(data.values)).reshape(1,-1)
    output=RF.predict(data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=RF.predict(final_input)[0]
    return render_template("home.html",prediction_text="Credit score is {}".format(output))

if __name__ =="__main__":
    app.run(debug=True)
