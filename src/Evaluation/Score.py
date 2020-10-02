import rouge
import os.path
import os
import shutil
import pandas as pd

class Evaluation(object):
    def __init__(self, sampleNo,Center=False,Theme=True):
        self.sampleNo=sampleNo
        self.Theme=Theme
        self.Center=Center

    def createReference(self):
        references=[]
        if(self.Theme):
            referencePath = './Theme/'
            filepathRef = os.path.join(referencePath, "theme"+str(self.sampleNo)+".txt")

            file = open(filepathRef, "r")
            r = file.read()
            references.append(r)
            file.close()
        if(self.Center):
            referencePath = './Center/'
            filepathRef = os.path.join(referencePath, "center"+str(self.sampleNo)+".txt")
            file = open(filepathRef, "r")
            r = file.read()
            references.append(r)
            file.close()
        return references

    def createHypothesis(self):
        hypothesis=[]
        hypothesisPath = './Output/'
        filepathHypo = os.path.join(hypothesisPath, "sample"+str(self.sampleNo)+".txt")
        file = open(filepathHypo, "r")
        r = file.read()
        hypothesis.append(r)
        file.close()
        return hypothesis

    def getRouge(self, hypothesis, references):

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                max_n=3,
                                limit_length=True,
                                length_limit=100,
                                length_limit_type='words',
                                apply_avg='Avg',
                                apply_best=False,
                                alpha=0.5,  # Default F1_score
                                weight_factor=1.2,
                                stemming=True)
        scores = evaluator.get_scores(hypothesis, references)

        # preparing a list of scores
        metrics = ['rouge-1', 'rouge-2', 'rouge-3', 'rouge-l']

        listScore = []
        for metric in metrics:
            tempList = []
            tempList.append(metric)
            tempList.append(round(scores[metric]['p'], 3))
            tempList.append(round(scores[metric]['r'], 3))
            tempList.append(round(scores[metric]['f'], 3))
            listScore.append(tempList)
        return listScore

    def destroyDirectory(self):
        shutil.rmtree('./Theme/')
        shutil.rmtree('./Center/')
        shutil.rmtree('./Output/')


    def getScoreTableIntersect(self):
        df = pd.DataFrame(columns=['','Precision','Recall','F1'])
        hypothesis=self.createHypothesis()
        reference=self.createReference()
        score=self.getRouge(hypothesis,reference)

        for i in range(len(score)):
            df.loc[i]=score[i]
        self.destroyDirectory()
        return df

    def destroyDirectoryDif(self):
        shutil.rmtree('./Left/')
        shutil.rmtree('./Right/')
        shutil.rmtree('./Theme/')
        shutil.rmtree('./JMDifLeftBias/')
        shutil.rmtree('./JMDifRightBias/')

    def createReferenceTheme(self):
        references=[]
        referencePath = './Theme/'
        filepathRef = os.path.join(referencePath, "theme"+str(self.sampleNo)+".txt")
        file = open(filepathRef, "r")
        r = file.read()
        references.append(r)
        file.close()
        return references

    def createReferenceRight(self):
        references=[]
        referencePath = './Right/'
        filepathRef = os.path.join(referencePath, "right"+str(self.sampleNo)+".txt")
        file = open(filepathRef, "r")
        r = file.read()
        references.append(r)
        file.close()
        return references

    def createReferenceLeft(self):
        references=[]
        referencePath = './Left/'
        filepathRef = os.path.join(referencePath, "left"+str(self.sampleNo)+".txt")
        file = open(filepathRef, "r")
        r = file.read()
        references.append(r)
        file.close()
        return references

    def createHypothesisRight(self):
        hypothesis=[]
        hypothesisPath = './JMDifRightBias/'
        filepathHypo = os.path.join(hypothesisPath, "sample"+str(self.sampleNo)+".txt")
        file = open(filepathHypo, "r")
        r = file.read()
        hypothesis.append(r)
        file.close()
        return hypothesis

    def createHypothesisLeft(self):
        hypothesis=[]
        hypothesisPath = './JMDifLeftBias/'
        filepathHypo = os.path.join(hypothesisPath, "sample"+str(self.sampleNo)+".txt")
        file = open(filepathHypo, "r")
        r = file.read()
        hypothesis.append(r)
        file.close()
        return hypothesis

    def convDF(self, score1, score2):

        metrics = ['rouge-1', 'rouge-2', 'rouge-3', 'rouge-l']
        ScoreMul = []
        i=0
        for metric in metrics:
            tempList = []
            tempList.append(metric)

            #precision
            tempList.append(round(score1[i][1]*score2[i][1], 3))

            #recall
            tempList.append(round(score1[i][2]*score2[i][2], 3))

            #f1
            tempList.append(round(score1[i][3]*score2[i][3], 3))

            ScoreMul.append(tempList)
            i+=1

        #print(ScoreMul)
        df = pd.DataFrame(columns=['','Precision','Recall','F1'])
        for i in range(len(ScoreMul)):
            df.loc[i]=ScoreMul[i]
        return df

    def getScoreDifferencefromRight(self):
        rightBias=self.createHypothesisRight()
        theme=self.createReferenceTheme()
        left=self.createReferenceLeft()
        score1=self.getRouge(rightBias,theme)
        score2=self.getRouge(rightBias,left)
        #print(score1)
        #print(score2)
        return self.convDF(score1, score2)

    def getScoreDifferencefromLeft(self):
        leftBias=self.createHypothesisLeft()
        theme=self.createReferenceTheme()
        right=self.createReferenceRight()
        score1=self.getRouge(leftBias,theme)
        score2=self.getRouge(leftBias,right)
        #print(score1)
        #print(score2)
        return self.convDF(score1, score2)

    def getScoreTableDifference(self):

        print("ScoreTable: Right TextDifference Left")
        print(self.getScoreDifferencefromRight().to_string(index=False))

        print("\n\n")
        print("ScoreTable: Left TextDifference Right")
        print(self.getScoreDifferencefromLeft().to_string(index=False))

        self.destroyDirectoryDif()
