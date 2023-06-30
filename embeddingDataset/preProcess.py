import numpy as np
import pandas as pd
import string
import pandas as pd


PATH = "twcs.csv"


def preProcessData(PATH):
    data = pd.read_csv(PATH)
    textColumn = data[['text']].astype(str)
    textColumn['text'] = textColumn['text'].str.lower()

    removePunc = string.punctuation

    def removePunctuation(text):
        return text.translate(str.maketrans('', '', removePunc))

    textColumn['text'] = textColumn['text'].apply(
        lambda text: removePunctuation(text))

    return textColumn


embeddingData = preProcessData(PATH)
