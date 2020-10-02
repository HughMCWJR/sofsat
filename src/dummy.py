import pandas as pd
import random
from src.Intersection.Summary import *
from src.Evaluation.Score import *
from src.Difference.difference import *

def concat(left, right):
    out = left + right
    if len(out) > 500:
        out = out[:499] + "..."
    return out


def intersection(sample_num, output_length, df, df_idf, df_T):
    sm = Summary_Class(sample_num, df, df_idf, df_T)
    result = sm.intersection(length=output_length)
    result = result.split('\n')

    ev=Evaluation(sample_num,Center=True, Theme=False)
    df = ev.getScoreTableIntersect()
    table = df.values.tolist()

    for i in range(len(result)):
        result[i] = result[i].split()

    target_middle, target_left, target_right = sm.focusedWord()

    return result, table, target_middle, target_left, target_right

def difference(sample_num, output_length, df, df_idf, df_T):
    sm = Summary_Class_Difference(sample_num, df, df_idf, df_T)
    right_bias = sm.getRightBias(length=output_length).split('\n')
    left_bias = sm.getLeftBias(length=output_length).split('\n')

    for i in range(len(right_bias)):
        right_bias[i] = right_bias[i].split()
    for i in range(len(left_bias)):
        left_bias[i] = left_bias[i].split()

    ev = Evaluation(sample_num)
    right_scores = ev.getScoreDifferencefromRight().values.tolist()
    left_scores = ev.getScoreDifferencefromLeft().values.tolist()

    target_right = sm.focusWordFromRight()
    target_left = sm.focusWordFromLeft()

    return right_bias, left_bias, right_scores, left_scores, target_right, target_left


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
