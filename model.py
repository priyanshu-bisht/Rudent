import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import tokenizer
import pickle

df = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1', names=["target", "ids", "date", "flag", "user", "text"])
df = df[['target','text']]
df['target'] = df['target'].replace(4,1)

t = tokenizer(df['text'].values)
t.vocab()
t.remf(10,max(t.v.values()))
seqtext = t.sen2seq(t.sanatize(df['text'].values))
lst = []
for x in seqtext:
    lst.append(len(x))
padseq = t.pad(seqtext, max(lst))

X_train, X_test, y_train, y_test = train_test_split(np.array(padseq, dtype=np.int32), df['target'].values, test_size=0.01, random_state=0)

def makemodel():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(41000, 32, input_length=(max(lst))),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l1'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l1'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()
    return model
model = makemodel()

num_epochs = 5
model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), batch_size=64)

model.save('models/rudent.h5')
pickle.dump(t, open("models/tokenizer.pickle","wb"))

