import os.path
import os
import shutil
import pandas as pd

from datasets import load_dataset, load_metric
metric = load_metric('rouge')

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

    def getScore(self, hypothesis, reference):

        metric.add_batch(predictions=hypothesis, references=reference )
        score = metric.compute(rouge_types=["rouge1", "rouge2", "rouge3",  "rougeLsum"],
                                        use_agregator=True, use_stemmer=True)
        return score

    def getRouge(self, hypothesis, references):

        result = self.getScore(hypothesis, references)

        scoreFinal = []
        for k,v in result.items():
            listScore = []
            listScore.append(k)
            listScore.append(round(v.mid.precision * 100, 2))
            listScore.append(round(v.mid.recall * 100, 2))
            listScore.append(round(v.mid.fmeasure * 100, 2))
            scoreFinal.append(listScore)
        return scoreFinal

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

