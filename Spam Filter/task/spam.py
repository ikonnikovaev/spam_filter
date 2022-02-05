# write your code here
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

pd.options.mode.chained_assignment = None
en_sm_model = spacy.load("en_core_web_sm")

data = pd.read_csv('spam.csv', encoding='iso-8859-1')
df = data.iloc[:, :2]
df.columns = ['Target', 'SMS']
for i in df.index:
    txt = df.loc[i, 'SMS'].lower()
    #print(txt)
    doc = en_sm_model(txt)
    words = [token.lemma_ for token in doc if not token.is_punct]
    for k in range(len(words)):
        if words[k].isalpha():
            pass
        else:
            words[k] = 'aanumbers'
    lemmas = [w for w in words if not w in STOP_WORDS and not len(w) == 1]

    txt = " ".join(lemmas)
    #print(txt)
    df.loc[i, 'SMS'] = txt

pd.options.display.max_columns = df.shape[1]
pd.options.display.max_rows = df.shape[0]
print(df.iloc[:200, :])
#print(df.Target.value_counts())

