import numpy as np 
import pandas as pd
from gensim.models.fasttext import FastText

# use example set of texts
from gensim.test.utils import common_texts

#build the model and train
#use vector size 4 for the example
model = FastText(size=4, window=3, min_count=1)  # instantiate
model.build_vocab(sentences=common_texts)
model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)  # train

model.wv['computer']

#produce a list of word embeddings for the example
veclist = []
for i in common_texts[0:len(common_texts)]:
    veclist.append(model.wv[i])
print(veclist)
    