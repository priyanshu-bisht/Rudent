import tensorflow as tf
import numpy as np
import pickle
model = tf.keras.models.load_model('models/rudent.h5')
t = pickle.load(open('models/tokenizer.pickle', 'rb'))

def rudent(sen):
    tsens = t.pad(t.sen2seq(t.sanatize([sen])),41)
    classes = ['Rude', 'Not Rude']
    return classes[int(np.round(model.predict(tsens)))]

sentence = input('> Say Something: ')
print(rudent(sentence))
