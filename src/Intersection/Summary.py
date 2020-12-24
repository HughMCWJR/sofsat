import re
import pandas as pd
import os.path
import time
import json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import math
import string
from nltk.corpus import stopwords
# Computing TfIdf using TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

myStopWords = ['thi', 'theyv', 'theyr', 'arent', 'doesnt',  'whi', 'wa', 'hi', 'ha', 'dont', 'made', 'one', 'seemed', 'something', 'using', 'whereupon', 'last',
               'until', 'did', 'wherein', 'fifty', 'seem', 'above', 'show', 'sometimes', 'here', 'five', 'formerly', 'noone',
               'once', 'top', 'twenty', 'beside', 'afterwards', 'but', 'beyond', 'wherever', 'back', 'every', 'me', 'do', 'how', 'at',
               'becoming', 'hundred', 'below', 'indeed', 'much', 'name', 'even', 'somewhere', 'others', 'out', 'first', 'since', 'six',
               'else', 'be', 'say', 'make', 'few', 'whither', 'why', 'will', 'whereby', 'the', 'amongst', 'herein', 'three', 'whereas',
               'yours', 'cannot', 'where', 'own', 'go', 'front', 'of', 'thereafter', 'its', 'under', 'there', 'well', 'hers', 'move',
               'have', 'while', 'are', 'hereby', 'elsewhere', 'fifteen', 'whose', 'she', 'although', 'both', 'around', 'with', 'behind',
               'however', 'only', 'about', 'along', 'hereafter', 'nobody', 'does', 'now', 'other', 'please', 'ten', 'was', 'toward', 'not',
               'often', 'so', 'himself', 'whenever', 'regarding', 'thru', 'otherwise', 'my', 'seems', 'perhaps', 'never', 'rather', 'twelve',
               'which', 'some', 'through', 'somehow', 'nor', 'anyway', 'ever', 'done', 'into', 'nothing', 'bottom', 'just', 'per', 'sixty', 'her',
               'therefore', 'us', 'take', 'whom', 'always', 'further', 'over', 'might', 'this', 'or', 'our', 'thereby', 'from', 'latter', 'by', 'neither',
               'another', 'besides', 'these', 'see', 'without', 'no', 'each', 'has', 'put', 'may', 'what', 'empty', 'yet', 'least', 'myself', 'themselves',
               'none', 'down', 'get', 'keep', 'against', 'all', 'within', 'still', 'as', 'therein', 'someone', 'eleven', 'unless', 'nowhere', 'upon', 'it',
               'than', 'after', 'whoever', 'already', 'between', 'almost', 'four', 'him', 'in', 'before', 'alone', 'enough', 'on', 'to', 'those', 'also',
               'became', 'up', 'whereafter', 'eight', 'been', 'except', 'off', 'nevertheless', 'side', 'beforehand', 'should', 'anyone', 'an', 'for', 'due',
               'mine', 'ourselves', 'such', 'anything', 'thus', 'across', 'third', 'am', 'onto', 'anyhow', 'you', 'throughout', 'really', 'everything', 'serious',
               'less', 'that', 'your', 'had', 'more', 'two', 'his', 'among', 'during', 're', 'would', 'hereupon', 'hence', 'next', 'ca', 'ours', 'and', 'he', 'whole', 'itself', 'whence', 'thence', 'they', 'whatever', 'yourself', 'very', 'seeming', 'we', 'herself', 'being', 'sometime', 'former', 'too', 'if', 'any', 'again', 'meanwhile', 'part', 'everyone', 'various',
               'nine', 'them', 'several', 'then', 'via', 'must', 'latterly', 'is', 'become', 'becomes', 'either', 'give', 'quite', 'used', 'a', 'everywhere', 'who', 'were', 'when', 'doing', 'their', 'anywhere', 'though', 'together', 'namely', 'forty', 'mostly', 'towards', 'whether', 'full', 'i', 'thereupon', 'could', 'same', 'most', 'because', 'can', 'yourselves', 'amount', 'many', 'moreover', 'call']


class Summary_Class(object):
    def __init__(self, sample_no, df, df_idf, df_T):
        self.total_doc = 1475
        self.sample_no = sample_no
        self.sw = stopwords.words('english') + myStopWords
        self.df = df
        self.df_idf = df_idf

        self.df_T = df_T
        self.setReferenceTheme()
        self.setReferenceCenter()

        self.Doc1 = self.df_T['right-context'][sample_no]
        self.Doc2 = self.df_T['left-context'][sample_no]
        self.file = self.Doc1+' '+self.Doc2
        self.linePerTF = self.set_TF(self.file)

        self.Query = self.setQuery(self.Doc1, self.Doc2)[0]

    # Set a TF for each line
    def set_TF(self, file):
        Lines = tokenize.sent_tokenize(file)
        df = pd.DataFrame(columns=['line', 'TF_Dictionary', 'Line_Length'])
        i = 0
        for line in Lines:
            # line = preprocess_round1(line)
            line = preprocess_round2(line)
            d = {}
            words = tokenize.word_tokenize(line)
            words = [PorterStemmer().stem(w) for w in words]

            for word in words:
                try:
                    d[word] += 1
                except KeyError:
                    d[word] = 1

            # only keep sentence with words >=5
            if (len(words) >= 5):
                df.loc[i] = np.array([line, d, len(words)], dtype=object)
                i = i+1

        return df

    def setQuery(self, Doc1, Doc2):
        # Doc1 = preprocess_round1(Doc1)
        Doc1 = preprocess_round2(Doc1)

        # Doc2 = preprocess_round1(Doc2)
        Doc2 = preprocess_round2(Doc2)

        dic_Doc1 = {}
        words = tokenize.word_tokenize(Doc1)
        words = [PorterStemmer().stem(w) for w in words]

        for word in words:
            try:
                dic_Doc1[word] += 1
            except KeyError:
                dic_Doc1[word] = 1

        # print(dic_Doc1)

        dic_Doc2 = {}
        words = tokenize.word_tokenize(Doc2)
        words = [PorterStemmer().stem(w) for w in words]

        for word in words:
            try:
                dic_Doc2[word] += 1
            except KeyError:
                dic_Doc2[word] = 1
        # print(dic_Doc2)

        query_dic = {}
        self.query_len = 0
        for w in dic_Doc1:
            if w in dic_Doc2 and w not in self.sw:
                query_dic[w] = (2*dic_Doc1[w]*dic_Doc2[w]) / \
                    (dic_Doc1[w]+dic_Doc2[w])
                #query_dic[w] = np.log(dic_Doc1[w])+np.log(dic_Doc2[w])
                self.query_len += query_dic[w]
        return query_dic, dic_Doc1, dic_Doc2

    def JMScore(self, Query, Doc, lambda_, d_len):
        if (len(Doc) == 0) or (len(Query) == 0):
            return -math.inf
        # Query and Doc both in Dictionary format
        else:
            score = 0
            for w in Query:
                if w in Doc and w in self.df_idf.index:
                    c_w_q = Query[w]
                    c_w_d = Doc[w]
                    IDF = self.df_idf.loc[w, "idf_weights"]
                    in_log = ((1-lambda_)*c_w_d*IDF)/(lambda_*d_len)
                    score += c_w_q*np.log(1+in_log)
            return score

    def DPScore(self, Query, Doc, mu, d_len):

        if (len(Doc) == 0) or (len(Query) == 0):
            return -math.inf
        # Query and Doc both in Dictionary format
        else:
            score = 0
            for w in Query:
                if w in Doc and w in self.df_idf.index:
                    c_w_q = Query[w]
                    c_w_d = Doc[w]
                    IDF = self.df_idf.loc[w, "idf_weights"]
                    in_log = (c_w_d*IDF)/mu
                    score += c_w_q*np.log(1+in_log)
            score = score + np.log(self.query_len)*np.log(mu/(mu+d_len))
            return score

    def Summary(self, JM, lambda_, DP, mu, set):
        sentence_count = self.linePerTF.shape[0]

        # JMScore for each line
        JM_Score_list = [0] * sentence_count
        DP_Score_list = [0] * sentence_count

        for i in range(sentence_count):
            Line_Dict = self.linePerTF.loc[i]['TF_Dictionary']
            # print(Line_Dict)
            JM_Score_list[i] = self.JMScore(
                Query=self.Query, Doc=Line_Dict, lambda_=lambda_, d_len=self.linePerTF.loc[i]['Line_Length'])
            DP_Score_list[i] = self.DPScore(
                Query=self.Query, Doc=Line_Dict, mu=mu, d_len=self.linePerTF.loc[i]['Line_Length'])

        self.linePerTF['JM Score'] = JM_Score_list
        self.linePerTF['DP Score'] = DP_Score_list

        # print(self.linePerTF.head(5))

        JMSummary = self.linePerTF.sort_values(
            by='JM Score', ascending=False)

        DPSummary = self.linePerTF.sort_values(
            by='DP Score', ascending=False)

        if(JM == 1):
            #print(JMSummary.head(3))
            # Writing to file
            save_path_JM = './Output/'
            if not os.path.exists(save_path_JM):
                os.mkdir(save_path_JM)
            filename = os.path.join(
                save_path_JM, "sample"+str(self.sample_no)+".txt")
            file1 = open(filename, "w")
            for i in range(self.length):
                file1.write(JMSummary.iloc[i]['line']+"\n")
            file1.close()
            #print("JM Summary of", self.sample_no, "complete")

        if(DP == 1):
            #print(DPSummary.head(3))

            # Writing to file
            save_path_DP = './DP'+str(set)+'/'
            if not os.path.exists(save_path_DP):
                os.mkdir(save_path_DP)
            filename = os.path.join(
                save_path_DP, "DPSummary"+str(self.sample_no)+".txt")
            file1 = open(filename, "w")
            for i in range(sentence_count):
                file1.write(DPSummary.iloc[i]['line']+"\n")
            file1.close()
            #print("DP Summary of", self.sample_no, "complete")

        return JMSummary, DPSummary

    def setReferenceTheme(self):
        ref_path = './Theme/'
        if not os.path.exists(ref_path):
            os.mkdir(ref_path)
        filename = os.path.join(ref_path, "theme"+str(self.sample_no)+".txt")
        file1 = open(filename, "w")
        Doc = self.df_T['theme-description'][self.sample_no]
        Doc = preprocess_round1(Doc)
        Doc = preprocess_round2(Doc)

        file1.write(Doc)
        file1.close()

    def setReferenceCenter(self):
        ref_path = './Center/'
        if not os.path.exists(ref_path):
            os.mkdir(ref_path)
        filename = os.path.join(ref_path, "center"+str(self.sample_no)+".txt")
        file1 = open(filename, "w")

        Doc = self.df_T['center-context'][self.sample_no]
        Doc = preprocess_round1(Doc)
        Doc = preprocess_round2(Doc)

        file1.write(Doc)
        file1.close()

    def intersection(self, length=3):
        self.length=length
        JM = self.Summary(JM=1, lambda_=0.2, DP=0, mu=1000, set=0)

        readPath = './Output/'
        filepathRead = os.path.join(readPath, "sample"+str(self.sample_no)+".txt")
        fileRead = open(filepathRead, "r")

        allLine=" "
        for line in range(length):
            line = fileRead.readline()
            allLine+=line
        return allLine

    def focusedWord(self):
        middle, right, left = self.setQuery(self.Doc1, self.Doc2)

        word = sorted(middle, reverse=True)
        left_words = set(sorted(left, reverse=True))
        right_words = set(sorted(right, reverse=True))

        left_words = left_words - set(word)
        right_words = right_words - set(word)

        both = left_words & right_words
        left_words = left_words - set(both)
        right_words = right_words - set(both)

        left_words = left_words - set(self.sw)
        right_words = right_words - set(self.sw)

        return word, list(left_words), list(right_words)


def dataRead_func():
    openfile = open('static/data/data.json')
    jsondata = json.load(openfile)
    df = pd.DataFrame(jsondata)
    openfile.close()
    df_T = df.transpose()
    return df_T

def set_idf(df, total_doc):
    Docs = []

    for sample_no in range(total_doc):
        Doc = df['right-context'][sample_no]+ df['left-context'][sample_no] + \
            df['center-context'][sample_no]+df['theme-description'][sample_no] + \
            df['right-head'][sample_no]+df['left-head'][sample_no] + \
            df['center-head'][sample_no] + \
            df['theme'][sample_no]

        Doc = preprocess_round1(Doc)
        Doc = preprocess_round2(Doc)
        Docs.append(Doc)

    cv = CountVectorizer()

    word_count_vector = cv.fit_transform(Docs)
    # print(word_count_vector.toarray())
    # print(word_count_vector.shape)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    # idf values
    df_idf = pd.DataFrame(
        tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])

    return df_idf

def preprocess_round1(Doc):
    Doc = Doc.lower()

    # remove punctuation
    Doc = re.sub(re.escape(string.punctuation), "", Doc)

    # remove words containing digits
    Doc = re.sub('\w*\d\w*', '', Doc)

    # remove newlines
    Doc = re.sub('\n', '', Doc)

    # remove text in square brackets
    Doc = re.sub('\[.*?\]', '', Doc)

    # remove additional punctuation manually
    Doc = re.sub('\'', '', Doc)
    Doc = re.sub('\"', '', Doc)
    Doc = re.sub('\_', '', Doc)
    Doc = re.sub('\,', '', Doc)
    Doc = re.sub('\-', '', Doc)
    Doc = re.sub('\(', '', Doc)
    Doc = re.sub('\)', '', Doc)
    Doc = re.sub('\.', '', Doc)

    # removing single letters
    Doc = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', Doc)

    return Doc

def preprocess_round2(Doc):
    return ' '.join(word for word in Doc.split() if len(word) > 1)

def writeCleanSample():
    df_clean = pd.DataFrame(columns=['theme', 'theme-description', 'right-head',
                                     'right-context', 'center-head', 'center-context', 'left-head', 'left-context'])
    openfile = open('static/data/data.json')
    jsondata = json.load(openfile)
    df = pd.DataFrame(jsondata)
    openfile.close()
    df_T = df.transpose()

    samples = 1475
    count = 0
    i = 0
    for sample in range(0, samples):
        if(len(df_T['theme-description'][sample].split(" ")) > 20):
            data = df_T.iloc[sample]
            count += 1
            df_clean.loc[i] = [data['theme'], data['theme-description'], data['right-head'], data['right-context'],
                               data['center-head'], data['center-context'], data['left-head'], data['left-context']]
            i += 1
    #print("Clean item: ", count)
    return df_clean
