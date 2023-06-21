import json
import pickle
from transformers import BertTokenizer,BertForSequenceClassification
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np

import torch

app=Flask(__name__)
## Load the model
model_identifier = "zaid683/my_model_683"
#model=BertForSequenceClassification.from_pretrained('my_model',num_labels=6)
model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=6)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    token = BertTokenizer.from_pretrained('distilbert-base-uncased')
    inputs = token(data, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = predictions.detach().numpy()
    pred = pred[0]
    highest_index = np.argmax(pred)
    if highest_index == 0:
        return jsonify("sadness")
    elif highest_index == 1:
        return jsonify("anger")
    elif highest_index == 2:
        return jsonify("love")
    elif highest_index == 3:
        return jsonify("surprise")
    elif highest_index == 4:
        return jsonify("fear")
    else:
        return jsonify("joy")


@app.route('/predict',methods=['POST'])
def predict():
    data = list(request.form.values())
    token = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = token(data, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = predictions.detach().numpy()
    pred = pred[0]
    highest_index = np.argmax(pred)
    if highest_index == 0:
        values="sadness"
    elif highest_index == 1:
        values="anger"
    elif highest_index == 2:
        values="love"
    elif highest_index == 3:
        values="surprise"
    elif highest_index == 4:
        values ="fear"
    else:
        values="joy"
    return render_template("home.html",prediction_text="The sentence refers to  {}".format(values))



if __name__=="__main__":
    app.run(debug=True)
   
     
