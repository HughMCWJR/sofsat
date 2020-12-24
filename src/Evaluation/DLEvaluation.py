import os.path
import os
import shutil
import pandas as pd

from datasets import load_dataset, load_metric
metric = load_metric('rouge')

class DLEvaluation(object):
    def __init__(self, hypothesis, reference):
        self.hypothesis = []
        self.reference = []
        self.hypothesis.append(hypothesis)
        self.reference.append(reference)

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


    def getScoreTableDLIntersect(self):
        df = pd.DataFrame(columns=['','Precision','Recall','F1'])
        score=self.getRouge(self.hypothesis,self.reference)

        for i in range(len(score)):
            df.loc[i]=score[i]
        return df