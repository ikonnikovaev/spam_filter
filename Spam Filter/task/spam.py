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
df = df.sample(frac=1.0, random_state=43)
train_last_index = int(df.shape[0] * 0.8)
train_set = df[0:train_last_index]
test_set = df[train_last_index:]

# count spam and ham SMS
target_counts = train_set['Target'].value_counts()
nsms_ham = target_counts['ham']
nsms_spam = target_counts['spam']
p_ham = nsms_ham / (nsms_ham + nsms_spam)
p_spam = nsms_spam / (nsms_ham + nsms_spam)

# construct bag of words
bow = bag_of_words(train_set['SMS'])
data_with_bow = train_set.join(bow, how='outer')
data_with_bow.reset_index(drop=True, inplace=True)

# count occurrences of words in spam and ham
word_counts = data_with_bow.groupby(['Target']).sum()
# print(word_counts)

n_vocab = len(word_counts.columns)
nwords_ham = word_counts.loc['ham', :].sum()
nwords_spam = word_counts.loc['spam', :].sum()
alpha = 1

# calculate probabilities
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
# print(df_probabilities.iloc[:200, :])

predictions = pd.Series(index=test_set.index, dtype='string')


def predict(sms):
    condp_ham = p_ham
    condp_spam = p_spam
    for w in sms.split():
        if w in df_probabilities.index:
            condp_ham *= df_probabilities.loc[w, 'Ham Probability']
            condp_spam *= df_probabilities.loc[w, 'Spam Probability']
    if condp_ham > condp_spam:
        return 'ham'
    elif condp_ham < condp_spam:
        return 'spam'
    return 'unknown'
'''
for i in test_set.index:
    condp_ham = p_ham
    condp_spam = p_spam
    for w in test_set.loc[i, 'SMS'].split():
        if w in df_probabilities.index:
            condp_ham *= df_probabilities.loc[w, 'Ham Probability']
            condp_spam *= df_probabilities.loc[w, 'Spam Probability']
    if condp_ham > condp_spam:
        predictions[i] = 'ham'
    elif condp_ham < condp_spam:
        predictions[i] = 'spam'
    else:
        predictions[i] = 'unknown'

# print(predictions)
'''
df_predictions = test_set.copy()
df_predictions['Predicted'] = df_predictions['Target'].apply(predict)
df_predictions = df_predictions[['Predicted', 'Target']]
df_predictions.columns = ['Predicted', 'Actual']
# print(df_predictions.iloc[:50, :])

tp = 0
fp = 0
tn = 0
fn = 0

for i in df_predictions.index:
    if df_predictions.loc[i, 'Predicted'] == 'spam':
        if df_predictions.loc[i, 'Actual'] == 'spam':
            tp +=1
        elif df_predictions.loc[i, 'Actual'] == 'ham':
            fp +=1
    elif df_predictions.loc[i, 'Predicted'] == 'ham':
        if df_predictions.loc[i, 'Actual'] == 'spam':
            fn +=1
        elif df_predictions.loc[i, 'Actual'] == 'ham':
            tn +=1
#print(tp, tn, fp, fn)
metrics = {}
metrics['Accuracy'] = (tp + tn) / (tp + fp + tn + fn)
metrics['Recall'] = tp / (tp + fn)
metrics['Precision'] = tp / (tp + fp)
metrics['F1'] = 2 * metrics['Precision'] * metrics['Recall'] / (metrics['Precision'] + metrics['Recall'])

print(metrics)


