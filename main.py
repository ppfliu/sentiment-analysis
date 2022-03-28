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
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import pickle as pkl
import keras_bert as bert

from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer

from keras_bert.bert import get_custom_objects
from keras_bert.datasets import get_pretrained, PretrainedList

from keras_self_attention import SeqWeightedAttention

from data import load_data, load_bert_data
from model import build_model, build_bert_model

flags = tf.app.flags
flags.DEFINE_string(name='mode', default='train', help='set running mode: train, test')
flags.DEFINE_string(name='model_path', default='best_model.h5', help='filepath of the best model')
flags.DEFINE_string(name='model_name', default='cnn', help='model name')
flags.DEFINE_string(name='tokenizer_path', default='tokenizer.pkl', help='filepath of the tokenizer')
flags.DEFINE_string(name='language', default='cn', help='text language: cn, en')
flags.DEFINE_string(name='train_path', default='dataset/amazon_cn/train.csv', help='filepath of the train file')
flags.DEFINE_float(name='validation_split', default=0.2, help='split a fraction of the train set as validation set')
flags.DEFINE_string(name='test_path', default='dataset/amazon_cn/test.csv', help='filepath of the test file')
flags.DEFINE_integer(name='max_words', default=10000, help='max number of words in vocabulary')
flags.DEFINE_integer(name='max_length', default=150, help='max number of words in a document')
flags.DEFINE_integer(name='batch_size', default=8, help='number of examples in a batch')
flags.DEFINE_integer(name='num_classes', default=3, help='number of classes')
flags.DEFINE_integer(name='num_epochs', default=100, help='number of epochs')
flags.DEFINE_integer(name='patience', default=5, help='number of epochs without improvement for early stopping')


def main(args):
    mode = flags.FLAGS.mode
    tokenizer_path = flags.FLAGS.model_name + '_' + flags.FLAGS.tokenizer_path
    if mode == 'train':
        if flags.FLAGS.model_name.lower() == 'bert':
            if flags.FLAGS.language.lower() == 'cn':
                model_path = get_pretrained(PretrainedList.chinese_base)
            else:
                model_path = get_pretrained(PretrainedList.multi_cased_base)

            paths = bert.get_checkpoint_paths(model_path)
            token_dict = bert.load_vocabulary(paths.vocab)
            tokenizer = bert.Tokenizer(token_dict)

            x_train, y_train = load_bert_data(flags.FLAGS.train_path,
                                            flags.FLAGS.max_length,
                                            flags.FLAGS.num_classes,
                                            tokenizer)

            model = build_bert_model(paths.config, paths.checkpoint, flags.FLAGS.num_classes)
        else:
            tokenizer = Tokenizer(num_words=flags.FLAGS.max_words, char_level=True)
            x_train, y_train = load_data(flags.FLAGS.train_path,
                                         flags.FLAGS.max_length,
                                         flags.FLAGS.num_classes,
                                         tokenizer,
                                         fit_tokenizer=True)

            model = build_model(flags.FLAGS.max_words, flags.FLAGS.num_classes, flags.FLAGS.model_name)

        # simple early stopping
        early_stop = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=flags.FLAGS.patience)
        check_point = ModelCheckpoint(flags.FLAGS.model_path,
                                      monitor='val_acc',
                                      mode='max',
                                      verbose=1,
                                      save_best_only=True)

        print('Train...')
        model.fit(x_train,
                  y_train,
                  batch_size=flags.FLAGS.batch_size,
                  epochs=flags.FLAGS.num_epochs,
                  validation_split=flags.FLAGS.validation_split,
                  verbose=1,
                  callbacks=[early_stop, check_point])

        with open(tokenizer_path, 'wb') as tokenizer_file:
            pkl.dump(tokenizer, tokenizer_file, protocol=pkl.HIGHEST_PROTOCOL)

    elif mode == 'test':
        with open(tokenizer_path, 'rb') as tokenizer_file:
            tokenizer = pkl.load(tokenizer_file)
            if flags.FLAGS.model_name.lower() == 'bert':
                custom_objects = get_custom_objects()
                model = load_model(flags.FLAGS.model_path, custom_objects)
                x_test, y_test = load_bert_data(flags.FLAGS.test_path,
                                                flags.FLAGS.max_length,
                                                flags.FLAGS.num_classes,
                                                tokenizer)
            else:
                model = load_model(flags.FLAGS.model_path, custom_objects=SeqWeightedAttention.get_custom_objects())
                x_test, y_test = load_data(flags.FLAGS.test_path,
                                           flags.FLAGS.max_length,
                                           flags.FLAGS.num_classes,
                                           tokenizer,
                                           fit_tokenizer=False)

            score, acc = model.evaluate(x_test, y_test, batch_size=flags.FLAGS.batch_size)
            print('Test score: {:.4f}; accuracy: {:.4f}'.format(score, acc))

if __name__ == '__main__':
    tf.app.run()
