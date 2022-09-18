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
    data=data.values([1]).astype(float)
    #D1=np.array(list(data.values())).astype(float)
    print(np.array(list(data.values())[0]).reshape(1,-1))
    #newdata = [float(x) for x in data.values[0]] 
    output=RF.predict(data)
    print(output[0])
    return jsonify(output[0])

if __name__ =="__main__":
    app.run(debug=True)
