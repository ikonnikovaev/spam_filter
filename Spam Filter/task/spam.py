# write your code here
import re
import string
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# pd.options.mode.chained_assignment = None
en_sm_model = spacy.load("en_core_web_sm")
punct = string.punctuation.replace('', ' ').split()

def preprocess_text(txt):
    #print(txt)
    model = en_sm_model(txt.lower())
    lemmas = [token.lemma_ for token in model]
    no_punct = ' '.join([''.join([letter for letter in word if letter not in punct]) for word in lemmas])
    string_number = "aanumbers"
    no_numbers = [w if not re.search(r"[0-9]{1,}", w) else string_number for w in no_punct.split()]
    no_stop = [w for w in no_numbers if w not in STOP_WORDS]
    words = [w for w in no_stop if len(w) >= 2]
    '''
    words = [token.lemma_ for token in doc]
    for k in range(len(words)):
        if words[k].isalpha():
            pass
        else:
            words[k] = 'aanumbers'
    lemmas = [w for w in words if not w in STOP_WORDS and not len(w) == 1]
    '''
    return words


def make_vocabulary(texts):
    all_words = set()
    for i in texts.index:
        all_words.update(texts[i].split())
    return all_words

def bag_of_words(texts):
    all_words = list(make_vocabulary(texts))
    all_words.sort()
    #print(all_words)
    bow = pd.DataFrame(0, index=texts.index, columns=all_words)
    for i in bow.index:
        words = texts[i].split()
        for w in words:
            bow.loc[i, w] += 1
    #print(bow)
    return bow

data = pd.read_csv('spam.csv', encoding='iso-8859-1')
df = data.iloc[:, :2].copy()
df.columns = ['Target', 'SMS']

# preprocessing texts
for i in df.index:
    lemmas = preprocess_text(df.loc[i, 'SMS'])
    df.loc[i, 'SMS'] = " ".join(lemmas)

# splitting
'''
import numpy as np
np.random.seed(43)
random_index = np.random.permutation(df.index)
df_two = df.iloc[random_index].copy()
split = int(df_two.shape[0] * 0.8)
train = df_two[:split].copy()
test = df_two[split:].copy()
'''

df = df.sample(frac=1.0, random_state=43)
train_last_index = int(df.shape[0] * 0.8)
train_set = df[0:train_last_index]
test_set = df[train_last_index:]

bow = bag_of_words(train_set['SMS'])
data_with_bow = train_set.join(bow, how='outer')
data_with_bow.reset_index(drop=True, inplace=True)

word_counts = data_with_bow.groupby(['Target']).sum()
print(word_counts)

n_vocab = len(word_counts.columns)
nwords_ham = word_counts.loc['ham', :].sum()
nwords_spam = word_counts.loc['spam', :].sum()
alpha = 1

probabilities = {'Spam Probability': [], 'Ham Probability': []}
for w in word_counts.columns:
    p_spam = (word_counts.loc['spam', w] + alpha) / (nwords_spam + alpha * n_vocab)
    probabilities['Spam Probability'].append(p_spam)
    p_ham = (word_counts.loc['ham', w] + alpha) / (nwords_ham + alpha * n_vocab)
    probabilities['Ham Probability'].append(p_ham)

df_probabilities = pd.DataFrame(data=probabilities, index=word_counts.columns)

# output
pd.options.display.max_columns = 50
pd.options.display.max_rows = 200
print(df_probabilities.iloc[:200, :])

