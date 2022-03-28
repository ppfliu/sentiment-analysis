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

import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.utils import to_categorical

def load_data(file_path, max_len, num_classes, tokenizer, fit_tokenizer=False):

    df = pd.read_csv(file_path)
    docs = df['text'].tolist()
    labels = df['label'].tolist()

    if fit_tokenizer:
        tokenizer.fit_on_texts(docs)

    x_data = tokenizer.texts_to_sequences(docs)
    x_data = sequence.pad_sequences(x_data, maxlen=max_len)
    print('x_data shape:', x_data.shape)

    y_data = to_categorical(labels, num_classes)
    print('y_data shape:', y_data.shape)

    return x_data, y_data


def load_bert_data(file_path, max_len, num_classes, tokenizer):

    df = pd.read_csv(file_path)
    docs = df['text'].tolist()
    labels = df['label'].tolist()

    x_indices = []
    x_segments = []
    for doc_text in docs:
        indices, segments = tokenizer.encode(first=doc_text, max_len=max_len)
        x_indices.append(indices)
        x_segments.append(segments)

    x_data = [np.array(x_indices), np.array(x_segments)]
    y_data = to_categorical(labels, num_classes)

    return x_data, y_data
