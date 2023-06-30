import openai
from openai.embeddings_utils import get_embedding
import pandas as pd
from ast import literal_eval
import numpy as np



embedding = get_embedding(
    text="how can I help you today",
    engine="text-embedding-ada-002"
)

embeddedDataSet = pd.read_csv("customerServiceEmbeddings.csv")
embeddedDataSet['embeddings'] = embeddedDataSet['embeddings'].apply(
    literal_eval)
x = np.asarray(embeddedDataSet['embeddings'][0])
y = np.asarray(embedding)

print(x*y)
