import numpy as np
from sklearn.model_selection import train_test_split
import json
from sklearn.model_selection import StratifiedShuffleSplit



def read_data(path):
    return json.load(open(path)) 

def save_data(data,path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)

train_data = read_data('train.json')
print type(train_data)

selected_classes = ['italian','greek','french']

labels = [instance['cuisine'].encode("utf-8") for instance in train_data if instance['cuisine'].encode("utf-8") in selected_classes]
texts = [instance['ingredients'] for instance in train_data if instance['cuisine'].encode("utf-8") in selected_classes]
ids = [instance['id'] for instance in train_data if instance['cuisine'].encode("utf-8") in selected_classes]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.17, random_state=42)

for train_index, test_index in sss.split(texts, labels):
    print("TRAIN:%d, TEST:%d\n"%(len(train_index),len(test_index)))
    ratio = float(len(test_index))/float(len(train_index))
    print("RATIO %f"%ratio)

train_data_balanced = []
test_data_balanced = []

for idx in range(len(train_index)):
    instance = dict({'id':ids[idx], 'cuisine':labels[idx], 'ingredients':texts[idx]})
    train_data_balanced.append(instance)

print(train_index)
print("TRAIN: done %d instances"%idx)

for idx in range(len(test_index)):
    instance = dict({'id':ids[idx], 'cuisine':labels[idx], 'ingredients':texts[idx]})
    test_data_balanced.append(instance)

print(test_index)
print("TEST: done %d instances"%idx)

save_data(train_data_balanced,'train_data_homework.json')
save_data(test_data_balanced,'test_data_homework.json')


