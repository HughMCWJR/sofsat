from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename

import pandas as pd
import json
import os

from src.Intersection.Summary import *
from src.concatenate_docs import comb_inputs
import src.dummy as dummy
from rand import sum

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

df = dataRead_func()
df_idf = set_idf(df, 1475)
df_T = writeCleanSample()

# CSV file in which each article number has been assigned a specific topic
# Entry example of this csv: Congress 215
topics = pd.read_csv("static/data/NewsTopicsNew.csv")

election = topics.loc[topics['Topic'] == "Election"]
congress = topics.loc[topics['Topic'] == "Congress"]
foreign = topics.loc[topics['Topic'] == "Foreign"]
immigration = topics.loc[topics['Topic'] == "Immigration"]
pres = topics.loc[topics['Topic'] == "White House"]

@app.route("/", methods = ['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route("/own", methods=["POST"])
def own():
    left_head = request.form.get("left_head")
    right_head = request.form.get("right_head")
    left_body = request.form.get("left_body")
    right_body = request.form.get("right_body")
    result = dummy.concat(left_body, right_body)
    center_head = dummy.concat(left_head, right_head)
    return render_template("Intersect.html", left_head = left_head, right_head = right_head,
                            left_body = left_body, right_body = right_body, result = result,
                            center_head = center_head)

@app.route("/loaded", methods=["POST"])
def retrieve():
    
    left_head = request.form.get("left_head").split()
    right_head = request.form.get("right_head").split()

    left_body = request.form.get("left_body").split('\n')
    right_body = request.form.get("right_body").split('\n')

    for i in range(len(left_body)):
        left_body[i] = left_body[i].split()
    for i in range(len(right_body)):
        right_body[i] = right_body[i].split()

    center_head = dummy.concat(left_head, right_head)
    sample_num = int(request.form.get("sample_num"))
    output_length = 3
    model = request.form.get("model")
    result, table, target_middle, target_left, target_right = dummy.intersection(sample_num, output_length, df, df_idf, df_T)
    models = ['BERT-base', 'Mobile-BERT', 'DistilBERT', 'RoBERTa', 'XLNet', 'GPT2']

    if model == "Query-based":   
        return render_template("Intersect.html", left_head=left_head, right_head=right_head,
                                left_body=left_body, right_body=right_body, result=result,
                                center_head=center_head, scoreTable=table, target_left=target_left,
                                target_right=target_right, target_middle=target_middle)
    elif model in models:
        summary, stable = dummy.DLintersection(sample_num, output_length, model)
        return render_template("DLIntersect.html", left_head=left_head, right_head=right_head,
                                left_body=left_body, right_body=right_body, result=summary,
                                center_head=center_head, scoreTable=stable, target_left=target_left,
                                target_right=target_right, target_middle=target_middle)
    else:
        return str(model)

@app.route("/topicSelect", methods = ["POST"])
def topicSelect():
    topics = {
        'Elections': election,
        'The White House': pres,
        'Immigration': immigration,
        'Congress': congress,
        'Foreign Affairs': foreign
    }
    topic = request.form.get("topic")
    if topic in topics:
        db = topics[topic]
    else:
        db = None

    left_head, right_head, left_body, right_body, sample_num = dummy.loadRandomNew(df_T, db)

    return render_template("topicSelect.html", left_head = left_head, right_head = right_head,
                            left_body = left_body, right_body = right_body, sample_num = sample_num)

@app.route("/typeTextResult", methods = ['GET', 'POST'])
def typeTextResult():
    file_1 = request.form.get('file1')
    file_2 = request.form.get('file2')

    combined = dummy.concat(file_1, file_2)
    model = request.form.get('model')
    model_name = 'distilbert'

    f1 = open('raw_data/file1.txt', 'w')
    f1.write(combined)
    f1.close()

    # f2 = open('raw_data/file2.txt', 'w')
    # f2.write(file_2)
    # f2.close()

    if(model == 'BERT-base'):
        model_name = 'bertbase'
    elif(model == 'Mobile-BERT'):
        model_name = 'mobilebert'
    else:
        model_name = 'distilbert'

    print(model)    

    sum('raw_data/file1.txt', 'summary/file1.txt', model_name)
    # sum('raw_data/file2.txt', 'summary/file2.txt', model_name)

    summary_f1 = open('summary/file1.txt', 'r')
    summary_1 = summary_f1.read()

    # summary_f2 = open('summary/file2.txt', 'r')
    # summary_2 = summary_f2.read()

    return render_template('typeTextResult2.html',original_1 = file_1, original_2 = file_2, summary_1 = summary_1)

@app.route('/uploadTextResult', methods = ['GET', 'POST'])
def uploadTextResult():
    
    file1 = request.files['file1']
    file2 = request.files['file2']

    file1_data = file1.read()
    file2_data = file2.read()

    f1 = open('raw_data/file1.txt', 'w')
    f1.write(file1_data)
    f1.close()

    f2 = open('raw_data/file2.txt', 'w')
    f2.write(file2_data)
    f2.close()

    sum('raw_data/file1.txt', 'summary/file1.txt')
    sum('raw_data/file2.txt', 'summary/file2.txt')

    result_f1 = open('summary/file1.txt', 'r')
    result_1 = result_f1.read()

    result_f2 = open('summary/file2.txt', 'r')
    result_2 = result_f2.read()


    return render_template('uploadTextResult.html', file1 = result_1, file2 = result_2)

if __name__ == "__main__":
    app.run(debug=True)