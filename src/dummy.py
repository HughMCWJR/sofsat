import pandas as pd
import random
from src.Intersection.Summary import *
from src.Evaluation.Score import *
from src.Difference.difference import *
import os.path
from src.Evaluation.DLEvaluation import *
import nltk


def concat(left, right):
    out = left + right
    if len(out) > 500:
        out = out[:499] + "..."
    return out

def DLintersection(sample_num, output_length, model):
    referencePath = os.path.join(
        'static/data/DLOutput/Theme/', "theme"+str(sample_num)+".txt")
    refRead = open(referencePath, "r")
    reference = refRead.read()

    if model == "BERT-base":
        readPath = 'static/data/DLOutput/bertbase/'
        filepathRead = os.path.join(
        readPath, "summary"+str(sample_num)+".txt")

    elif model == "Mobile-BERT":
        readPath = 'static/data/DLOutput/mobilebert/'
        filepathRead = os.path.join(
        readPath, "summary"+str(sample_num)+".txt")      

    elif model == "DistilBERT":
        readPath = 'static/data/DLOutput/DistilBERT/'
        filepathRead = os.path.join(
        readPath, "summary"+str(sample_num)+".txt") 

    elif model == "RoBERTa":
        readPath = 'static/data/DLOutput/RoBERTa/'
        filepathRead = os.path.join(
        readPath, "summary"+str(sample_num)+".txt") 

    elif model == "XLNet":
        readPath = 'static/data/DLOutput/XLNet/'
        filepathRead = os.path.join(
        readPath, "summary"+str(sample_num)+".txt") 
    elif model == "GPT2":
        readPath = 'static/data/DLOutput/GPT2/'
        filepathRead = os.path.join(
        readPath, "summary"+str(sample_num)+".txt") 
    else:
        pass
    fileRead = open(filepathRead, "r")
    allLine=" "
    for line in range(output_length):
        line = fileRead.readline()
        allLine+=line
    result = allLine
    ev = DLEvaluation(hypothesis=result, reference=reference)
    df = ev.getScoreTableDLIntersect()
    score = df.values.tolist()
    
    result = result.split('\n')
    for i in range(len(result)):
        result[i] = result[i].split()
    return result, score

def intersection(sample_num, output_length, df, df_idf, df_T):
    sm = Summary_Class(sample_num, df, df_idf, df_T)
    result = sm.intersection(length=output_length)
    result = result.split('\n')

    ev=Evaluation(sample_num,Center=False, Theme=True)
    df = ev.getScoreTableIntersect()
    table = df.values.tolist()

    for i in range(len(result)):
        result[i] = result[i].split()

    target_middle, target_left, target_right = sm.focusedWord()


    return result, table, target_middle, target_left, target_right


def loadRandom(df):
    story = random.randrange(0, len(df)-1)

    left_head = df["left-head"][story]
    right_head = df["right-head"][story]
    left_body = df["left-context"][story]
    right_body = df["right-context"][story]
    result = df["center-context"][story]
    center_head = df["center-head"][story]

    if len(result) > 500:
        result = result[:499] + "..."

    return left_head, right_head, left_body, right_body, result, center_head

def loadRandomNew(df, db):
    if isinstance(db, pd.DataFrame):
        size = len(db)
        cur = db.to_numpy()
        lot = random.randint(0, size-1)
        sample_num = cur[lot][1]
        while sample_num > 425:
            lot = random.randint(0, size-1)
            sample_num = cur[lot][1]
    else:
        sample_num = random.randint(0, 425)

    left_head = df["left-head"][sample_num]
    right_head = df["right-head"][sample_num]
    left_body = df["left-context"][sample_num]
    right_body = df["right-context"][sample_num]


    return left_head, right_head, left_body, right_body, sample_num
