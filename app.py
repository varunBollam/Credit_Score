import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

#Loading model in VS code
app=Flask(__name__)
DTnewModel=pickle.load(open('DTnewmodel.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.rout('/predit_api',methods=['POST'])

def predit_api():
    data=request.json['data']
    print(data)
    #print(np.array(list(data.values())[0]).reshape(1,-1))
    output=DTnewModel.predit(data)
    print(output[0])
    return jsonify(output[0])

if __name__ =="__main__":
    app.run(debug=True)



