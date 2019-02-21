import numpy as np
import random

import json
def read_data(path):
    return json.load(open(path)) 

train_data = read_data('data/train_bss.json')
test_data = read_data('data/test_bss.json')

print("Number of training examples: %d"%len(train_data))
print("Number of test examples: %d"%len(test_data))

ntrain = len(train_data)
ntest = len(test_data)

### Explore the Structure of the Datasets
idx = random.randint(0, ntrain)
instance = train_data[idx] #Take the first training example
#print(type(instance)) #Discover this is a dictionary
keys = [key.encode("utf-8") for key in instance.keys()] #Collect all the Keys
print("Keys: %s"%str(keys))

print(" ")
print("Choosing a ramdom example ... \n")
print("ID & class: %s, %s"%(instance['id'],instance['cuisine']))
ingredients = ','.join(instance['ingredients'])
print(" ")
print("Ingredients: %s\n"% ingredients)

selected_classes = ['italian','greek','french']

labels_train = [instance['cuisine'].encode("utf-8") for instance in train_data if instance['cuisine'].encode("utf-8") in selected_classes]
print("Classes in the TRAIN set: "+','.join(np.unique(labels_train))+"\n")
print("Number of Classes in the TRAIN set: %d\n"%len(np.unique(labels_train)))

labels_test = [instance['cuisine'].encode("utf-8") for instance in test_data if instance['cuisine'].encode("utf-8") in selected_classes]
print("Classes in the TEST set: "+','.join(np.unique(labels_test))+"\n")
print("Number of Classes in the TEST set: %d\n"%len(np.unique(labels_test)))

texts_train = [' '.join(instance['ingredients']).lower() for instance in train_data if instance['cuisine'].encode("utf-8") in selected_classes]
labels_train = [instance['cuisine'].encode("utf-8") for instance in train_data if instance['cuisine'].encode("utf-8") in selected_classes]

from sklearn.feature_extraction.text import CountVectorizer

TF = CountVectorizer()
vocabulary = None

def represent_my_texts(texts, set_type):
    if set_type == 'train':
        X = TF.fit_transform(texts)
    else:
        X = TF.transform(texts)
    return X.astype('float16')

X_train = represent_my_texts(texts_train,'train')
print(X_train.shape)
 
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y_train = lb.fit_transform(labels_train)
print(np.unique(y_train))
print(len(y_train))
print(lb.inverse_transform(0)) #who is class 9?

texts_test = [' '.join(instance['ingredients']).lower() for instance in test_data if instance['cuisine'].encode("utf-8") in selected_classes]
labels_test = [instance['cuisine'].encode("utf-8") for instance in test_data if instance['cuisine'].encode("utf-8") in selected_classes]

X_test = represent_my_texts(texts_test,'test')
print(X_test.shape)
y_test = lb.fit_transform(labels_test)
print(len(y_test))

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import sys

Cval = float(sys.argv[1])
gammaval = float(sys.argv[2])

svm = SVC(C=Cval, # penalty parameter
                 kernel='rbf', # kernel type, rbf working fine here
                 degree=3, # default value
                 gamma=gammaval, # kernel coefficient
                 coef0=1, # change to 1 from default value of 0.0
                 shrinking=True, # using shrinking heuristics
                 tol=0.001, # stopping criterion tolerance 
                 probability=False, # no need to enable probability estimates
                 cache_size=200, # 200 MB cache size
                 class_weight=None, # all classes are treated equally 
                 verbose=False, # print the logs 
                 max_iter=-1, # no limit, let it run
                 decision_function_shape=None, # will use one vs rest explicitly 
                 random_state=None)


model = OneVsRestClassifier(svm, n_jobs=8)

model.fit(X_train, y_train)

score_train = model.score(X_train,y_train)
score_test = model.score(X_test,y_test)

print("Score RBF in the *TRAIN* set: %f"%score_train)
print("Score RBF in the *TEST* set: %f"%score_test)


from sklearn.svm import LinearSVC
lsvm = LinearSVC(C=Cval,random_state=0,tol=1e-5)
lmodel = OneVsRestClassifier(lsvm, n_jobs=8)
lmodel.fit(X_train, y_train)

lscore_train = lmodel.score(X_train,y_train)
lscore_test = lmodel.score(X_test,y_test)

print("Score LINEAR in the *TRAIN* set: %f"%lscore_train)
print("Score LINEAR in the *TEST* set: %f"%lscore_test)


##################################################################  
###################### TUNNING THE RBF ###########################
##################################################################  

from sklearn.model_selection import ParameterGrid
param_grid_rbf = {'reg': [0.001,0.01,0.1,1,10,100,1000], 'gamma': [0.001,0.01,0.1,1,2,5]}

list_scores_test = []
list_scores_train = []
list_params = []

for param_combination in ParameterGrid(param_grid_rbf):

	svm = SVC(C=param_combination['reg'], # penalty parameter
                 kernel='rbf', # kernel type, rbf working fine here
                 gamma=param_combination['gamma'], # kernel coefficient
                 coef0=1, # change to 1 from default value of 0.0
                 shrinking=True, # using shrinking heuristics
                 tol=0.001, # stopping criterion tolerance 
                 probability=False, # no need to enable probability estimates
                 cache_size=200, # 200 MB cache size
                 class_weight=None, # all classes are treated equally 
                 verbose=False, # print the logs 
                 max_iter=-1, # no limit, let it run
                 decision_function_shape=None, # will use one vs rest explicitly 
                 random_state=None)
	model = OneVsRestClassifier(svm, n_jobs=8)
	model.fit(X_train, y_train)
	score_train = model.score(X_train,y_train)
	score_test = model.score(X_test,y_test)
	list_scores_train.append(score_train)
	list_scores_test.append(score_test)
	list_params.append(param_combination)

results = open('tunning_rbf.csv','w')

sorted_idx = np.argsort(list_scores_test)
sorted_idx = sorted_idx[::-1]
list_scores_train = [list_scores_train[idx] for idx in sorted_idx]
list_scores_test = [list_scores_test[idx] for idx in sorted_idx]
list_params = [list_params[idx] for idx in sorted_idx]

for idx in range(len(list_scores_test)):
	params = list_params[idx]
	results.write("%.4f, %.4f, %.4f, %.4f\n"%(params['reg'],params['gamma'],list_scores_test[idx],list_scores_train[idx]))

results.close()

##################################################################  
###################### TUNNING THE LINEAR ########################
##################################################################  

param_grid_rbf = {'reg': [0.001,0.01,0.1,1,10,100,1000]}

list_scores_test = []
list_scores_train = []
list_params = []

for param_combination in ParameterGrid(param_grid_rbf):
	lsvm = LinearSVC(C=param_combination['reg'],random_state=0,tol=1e-5)
	lmodel = OneVsRestClassifier(lsvm, n_jobs=8)
	model.fit(X_train, y_train)
	score_train = model.score(X_train,y_train)
	score_test = model.score(X_test,y_test)
	list_scores_train.append(score_train)
	list_scores_test.append(score_test)
	list_params.append(param_combination)

results = open('tunning_linear.csv','w')

sorted_idx = np.argsort(list_scores_test)
sorted_idx = sorted_idx[::-1]
list_scores_train = [list_scores_train[idx] for idx in sorted_idx]
list_scores_test = [list_scores_test[idx] for idx in sorted_idx]
list_params = [list_params[idx] for idx in sorted_idx]

for idx in range(len(list_scores_test)):
	params = list_params[idx]
	results.write("%.4f, %.4f, %.4f\n"%(params['reg'],list_scores_test[idx],list_scores_train[idx]))

results.close()

##################################################################  
###################### TUNNING THE POLY ########################
##################################################################  

from sklearn.model_selection import ParameterGrid
param_grid_rbf = {'reg': [0.001,0.01,0.1,1,10,100,1000], 'degree': [1,2,3,4,5,6]}

list_scores_test = []
list_scores_train = []
list_params = []

for param_combination in ParameterGrid(param_grid_rbf):

	svm = SVC(C=param_combination['reg'], # penalty parameter
                 kernel='rbf', # kernel type, rbf working fine here
                 degree=param_combination['degree'], # kernel coefficient
                 coef0=1, # change to 1 from default value of 0.0
                 shrinking=True, # using shrinking heuristics
                 tol=0.001, # stopping criterion tolerance 
                 probability=False, # no need to enable probability estimates
                 cache_size=200, # 200 MB cache size
                 class_weight=None, # all classes are treated equally 
                 verbose=False, # print the logs 
                 max_iter=-1, # no limit, let it run
                 decision_function_shape=None, # will use one vs rest explicitly 
                 random_state=None)
	model = OneVsRestClassifier(svm, n_jobs=8)
	model.fit(X_train, y_train)
	score_train = model.score(X_train,y_train)
	score_test = model.score(X_test,y_test)
	list_scores_train.append(score_train)
	list_scores_test.append(score_test)
	list_params.append(param_combination)

results = open('tunning_poly.csv','w')

sorted_idx = np.argsort(list_scores_test)
sorted_idx = sorted_idx[::-1]
list_scores_train = [list_scores_train[idx] for idx in sorted_idx]
list_scores_test = [list_scores_test[idx] for idx in sorted_idx]
list_params = [list_params[idx] for idx in sorted_idx]

for idx in range(len(list_scores_test)):
	params = list_params[idx]
	results.write("%.4f, %.4f, %.4f, %.4f\n"%(params['reg'],params['degree'],list_scores_test[idx],list_scores_train[idx]))

results.close()
