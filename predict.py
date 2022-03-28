# Copyright (C) 2022 Pengfei Liu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import pickle as pkl
import numpy as np

from keras.models import load_model
from keras_bert.bert import get_custom_objects

flags = tf.app.flags
flags.DEFINE_string(name='model_path', default='best_model.h5', help='filepath of the best model')
flags.DEFINE_string(name='tokenizer_path', default='tokenizer.pkl', help='filepath of the tokenizer')
flags.DEFINE_string(name='doc_text', default='I like this movie very much!', help='document to be analyzed')
flags.DEFINE_integer(name='max_length', default=150, help='max number of words in a document')


def load_bert_data(doc_text, max_len, tokenizer):
    indices, segments = tokenizer.encode(first=doc_text, max_len=max_len)
    x_data = [np.array([indices]), np.array([segments])]
    return x_data


def main(args):
    labels = ['negative', 'positive', 'neutral']
    with open(flags.FLAGS.tokenizer_path, 'rb') as tokenizer_file:
        model = load_model(flags.FLAGS.model_path, get_custom_objects())
        x_test = load_bert_data(flags.FLAGS.doc_text, flags.FLAGS.max_length, pkl.load(tokenizer_file))

        y_pred = model.predict(x_test)[0]
        y_class = y_pred.argmax(axis=-1)
        print('scores: {}'.format(y_pred))
        print('class: {}'.format(y_class))

        result = {'sentiment': labels[y_class], 'scores': {lab: pred for lab, pred in zip(labels, y_pred)}}
        print(result)

if __name__ == '__main__':
    tf.app.run()

