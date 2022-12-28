import numpy as np
import cv2
import pickle
from flask import Flask,request,render_template
import os

# UPLOAD_FOLDER = './upload'
app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file1 = request.files['file']
        # path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        path = 'upload/' + file1.filename
        file1.save(path)
        temp = cv2.resize(cv2.imread(path,0),(256,256))
        res = model.predict(temp.reshape(1,-1))
        if res[0] == 0:
            return render_template('index.html',result='No Cancer')
        else:
            return render_template('index.html',result='Cancer')

if __name__ == '__main__':
    app.run(debug=True)