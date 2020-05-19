import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import tokenize  # tokenizer used when training TFIDF vectorizer
import pickle
import os

def add_question(question,predictions,sincere):
    if not os.path.exists('dataset.csv'):
        with open('dataset.csv','w') as file:
            file.write('question, toxic ,severe_toxic ,hate ,insult , obscene, threat, sincere\n')
    if os.path.exists('dataset.csv'):
        with open('dataset.csv','a') as file:
            file.write(f"""{question},{predictions['pred_toxic']},{predictions['pred_severe_toxic']},{predictions['pred_identity_hate']},{predictions['pred_insult']},{predictions['pred_obscene']},{predictions['pred_threat']},{sincere}\n""")
        return True
    else:
        return False

def analyse_message(msg):
    dict_preds = {}
    comment_term_doc = tfidf_model.transform([msg])
    dict_preds['pred_toxic'] = logistic_toxic_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_severe_toxic'] = logistic_severe_toxic_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_identity_hate'] = logistic_identity_hate_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_insult'] = logistic_insult_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_obscene'] = logistic_obscene_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_threat'] = logistic_threat_model.predict_proba(comment_term_doc)[:, 1][0]
    for k in dict_preds:
        perc = dict_preds[k] * 100
        dict_preds[k] = "{0:.2f}%".format(perc)
    return dict_preds

def analyse_sincerity(pred,threshold=30):
    sincere = True
    for k,v in pred.items():
        v = v.replace('%','')
        v = float(v)
        if v > threshold:
            sincere = False
            reason = k.split("_")[1]
        
    if sincere == False:
        return "insincere",reason
    else:
        return "sincere"




st.image('img.jpg',use_column_width=True)
st.title("Quora Insincere Question Classification")

st.write("Please write/paste some message or comment below")
message = st.text_area("enter text")
button = st.button("submit")    
with st.spinner("please wait, ML models are loading"):
    basepath = os.path.abspath(os.getcwd())

    with open(basepath + '/models/tfidf_vectorizer_train.pkl', 'rb') as tfidf_file:
            tfidf_model = pickle.load(tfidf_file)

    with open(basepath + '/models/logistic_toxic.pkl', 'rb') as logistic_toxic_file:
        logistic_toxic_model = pickle.load(logistic_toxic_file)

    with open(basepath + '/models/logistic_severe_toxic.pkl', 'rb') as logistic_severe_toxic_file:
        logistic_severe_toxic_model = pickle.load(logistic_severe_toxic_file)

    with open(basepath + '/models/logistic_identity_hate.pkl', 'rb') as logistic_identity_hate_file:
        logistic_identity_hate_model = pickle.load(logistic_identity_hate_file)

    with open(basepath + '/models/logistic_insult.pkl', 'rb') as logistic_insult_file:
        logistic_insult_model = pickle.load(logistic_insult_file)

    with open(basepath + '/models/logistic_obscene.pkl', 'rb') as logistic_obscene_file:
        logistic_obscene_model = pickle.load(logistic_obscene_file)

    with open(basepath + '/models/logistic_threat.pkl', 'rb') as logistic_threat_file:
        logistic_threat_model = pickle.load(logistic_threat_file)

checkraw = st.checkbox("view raw output")
if  button :
    if message:
        results = analyse_message(message)
        output = analyse_sincerity(results)
        if isinstance(output,tuple):
            prediction, reason = output
            st.info(f"the question above is {prediction}.")
            st.write(f"reason : the question is {reason}.")
        elif isinstance(output,str):
            st.info(f"the question above is {output}.")
        if checkraw:
            st.write(results)
        add_question(message,results,output)
    else:
        st.error("please enter a question asked on quora")


try:
    df = pd.read_csv("dataset.csv")
    if st.checkbox("view raw database"):
        st.write("database")
        st.write(df)
except:
    st.error("add some question for sicerity prediction")

op3 =st.checkbox("project info")
if op3:
    st.header("project team")
    st.write("Ashwani & Akash")