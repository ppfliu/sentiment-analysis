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

import sys
import csv

filename = sys.argv[1]

counter = {'pos':0, 'neu':0, 'neg': 0}
with open(filename, 'r') as csv_file:
    reader = csv.reader(csv_file)

    #skip header
    next(reader)

    for row in reader:
        label = int(row[1])
        if label == 0:
            counter['neg'] += 1
        elif label == 1:
            counter['pos'] += 1
        elif label == 2:
            counter['neu'] += 1
        else:
            print(row)
            break

print(counter)

