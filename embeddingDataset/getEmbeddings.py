import preProcess
import streamlit as st
import openai
from transformers import GPT2TokenizerFast
import numpy as np

# put  open.api_key="api key here"

data = preProcess.embeddingData.head(100)


def getEmbedding(data):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    data['n_tokens'] = data.text.apply(lambda x: len(tokenizer.encode(x)))

    data['embeddings'] = data.text.apply(lambda x: openai.Embedding.create(
        input=x, model='text-embedding-ada-002')['data'][0]['embedding'])

    return data


embeddedDataframe = getEmbedding(data)
embeddedDataframe.to_csv("../customerServiceEmbeddings.csv")
