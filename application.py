from flask import Flask, render_template, request
from src.Intersection.Summary import *
import pandas as pd
import json
import os
import src.dummy as dummy

app = Flask(__name__)

df = dataRead_func()
df_idf = set_idf(df, 1475)
df_T = writeCleanSample()

topics = pd.read_csv("static/data/NewsTopicsNew.csv")
election = topics.loc[topics['Topic'] == "Election"]
congress = topics.loc[topics['Topic'] == "Congress"]
foreign = topics.loc[topics['Topic'] == "Foreign"]
immigration = topics.loc[topics['Topic'] == "Immigration"]
pres = topics.loc[topics['Topic'] == "White House"]

@app.route("/")
def index():
    return render_template("Home.html")

@app.route("/load")
def load():
    return render_template("loadTopic.html")

@app.route("/about")
def about():
    return render_template("About.html")

@app.route("/own", methods=["POST"])
def own():
    left_head = request.form.get("left_head")
    right_head = request.form.get("right_head")
    left_body = request.form.get("left_body")
    right_body = request.form.get("right_body")
    result = dummy.concat(left_body, right_body)
    center_head = dummy.concat(left_head, right_head)
    return render_template("Intersect.html", left_head=left_head, right_head=right_head,
                            left_body=left_body, right_body=right_body, result=result,
                            center_head=center_head)

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

    if model == "Query-based":   
        return render_template("Intersect.html", left_head=left_head, right_head=right_head,
                                left_body=left_body, right_body=right_body, result=result,
                                center_head=center_head, scoreTable=table, target_left=target_left,
                                target_right=target_right, target_middle=target_middle)
    elif model == "BERT-base":
        summary, stable = dummy.DLintersection(sample_num, output_length, model)
        return render_template("DLIntersect.html", left_head=left_head, right_head=right_head,
                                left_body=left_body, right_body=right_body, result=summary,
                                center_head=center_head, scoreTable=stable, target_left=target_left,
                                target_right=target_right, target_middle=target_middle)
    elif model == "Mobile-BERT":
        summary, stable = dummy.DLintersection(sample_num, output_length, model)
        return render_template("DLIntersect.html", left_head=left_head, right_head=right_head,
                                left_body=left_body, right_body=right_body, result=summary,
                                center_head=center_head, scoreTable=stable, target_left=target_left,
                                target_right=target_right, target_middle=target_middle)
    elif model == "DistilBERT":
        summary, stable = dummy.DLintersection(sample_num, output_length, model)
        return render_template("DLIntersect.html", left_head=left_head, right_head=right_head,
                                left_body=left_body, right_body=right_body, result=summary,
                                center_head=center_head, scoreTable=stable, target_left=target_left,
                                target_right=target_right, target_middle=target_middle)
    elif model == "RoBERTa":
        summary, stable = dummy.DLintersection(sample_num, output_length, model)
        return render_template("DLIntersect.html", left_head=left_head, right_head=right_head,
                                left_body=left_body, right_body=right_body, result=summary,
                                center_head=center_head, scoreTable=stable, target_left=target_left,
                                target_right=target_right, target_middle=target_middle)
    elif model == "XLNet":
        summary, stable = dummy.DLintersection(sample_num, output_length, model)
        return render_template("DLIntersect.html", left_head=left_head, right_head=right_head,
                                left_body=left_body, right_body=right_body, result=summary,
                                center_head=center_head, scoreTable=stable, target_left=target_left,
                                target_right=target_right, target_middle=target_middle)
    elif model == "GPT2":
        summary, stable = dummy.DLintersection(sample_num, output_length, model)
        return render_template("DLIntersect.html", left_head=left_head, right_head=right_head,
                                left_body=left_body, right_body=right_body, result=summary,
                                center_head=center_head, scoreTable=stable, target_left=target_left,
                                target_right=target_right, target_middle=target_middle)
    else:
        return str(model)


@app.route("/LoadEdit", methods=["POST"])
def loadEdit():
    topic = request.form.get("topic")
    if topic == "Elections":
        db = election
    elif topic == "The White House":
        db = pres
    elif topic == "Immigration":
        db = immigration
    elif topic == "Congress":
        db = congress
    elif topic == "Foreign Affairs":
        db = foreign
    else:
        db = None


    left_head, right_head, left_body, right_body, sample_num = dummy.loadRandomNew(df_T, db)

    return render_template("LoadSample.html", left_head=left_head, right_head=right_head,
                            left_body=left_body, right_body=right_body, sample_num=sample_num)

if __name__ == "__main__":
    app.run(debug=True)
