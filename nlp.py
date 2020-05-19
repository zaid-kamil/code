import numpy as np
import os
import pickle


def load_model(path="models/model_tree.pk"):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

def load_scaler(path ='models/scaler.pk'):
    if os.path.exists(path):
        with open(path,'rb') as f:
            return pickle.load(f)

def predict(model,data):
    scaler = load_scaler()
    x = scaler.transform(data)
    results = model.predict(x)
    return results