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
# -*- coding: utf-8 -*-

from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam

from keras_bert import load_trained_model_from_checkpoint

from keras_self_attention import SeqWeightedAttention

def build_model(max_words, num_classes, model_name='lstm'):
    print('Build model: {}'.format(model_name))
    model = Sequential()
    model.add(Embedding(max_words, 128))

    if model_name.lower() == 'lstm':
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    elif model_name.lower() == 'lstmsa':
        model.add(LSTM(128, return_sequences=True))
        model.add(SeqWeightedAttention())
    elif model_name.lower() == 'cnn':
        model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
        model.add(GlobalMaxPooling1D())
    elif model_name.lower() == 'cnnsa':
        model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
        model.add(SeqWeightedAttention())
    else:
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()

    return model

def build_bert_model(config_path, checkpoint_path, num_classes):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

    for layer in bert_model.layers:
        layer.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(num_classes, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy'])
    model.summary()

    return model
