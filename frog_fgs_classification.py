#!/usr/bin/env python
# coding: utf-8

# ## 1. Multi-class and Multi-Label Classification Using Support Vector Machines

# ### Importing Libraries

import sklearn
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, hamming_loss, silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import collections

# To suppress warnings
warnings.filterwarnings('ignore')
# (a) Download the Anuran Calls (MFCCs) Data Set from GIT local repository

get_ipython().system(' git clone https://github.com/devikasathaye/Frogs-Families-Genus-Species-Classification')

# Choose 70% of the data randomly as the training set.
df_all = pd.read_csv('Frogs-Families-Genus-Species-Classification/Frogs_MFCCs.csv', sep=',', header=0, skiprows=0)
print("Entire dataset")
df_all

X = df_all.drop(columns=['Family','Genus','Species','RecordID'])
y = pd.DataFrame() # contains all the labels

y['Family'] = df_all['Family']
y['Genus'] = df_all['Genus']
y['Species'] = df_all['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# one dataframe for each label for train
y1_train = pd.DataFrame()
y2_train = pd.DataFrame()
y3_train = pd.DataFrame()

y1_train['Family'] = y_train['Family']
y2_train['Genus'] = y_train['Genus']
y3_train['Species'] = y_train['Species']

# one dataframe for each label for test
y1_test = pd.DataFrame()
y2_test = pd.DataFrame()
y3_test = pd.DataFrame()

y1_test['Family'] = y_test['Family']
y2_test['Genus'] = y_test['Genus']
y3_test['Species'] = y_test['Species']

print("X_train")
X_train

print("y_train")
y_train

print("y1_train")
y1_train

print("y2_train")
y2_train

print("y3_train")
y3_train

print("X_test")
X_test

print("y_test")
y_test

print("y1_test")
y1_test

print("y2_test")
y2_test

print("y3_test")
y3_test

# (b) One of the most important approaches to multi-class classification is to train a classifier for each label.

# 1(b) ii. Train a SVM for each of the labels, using Gaussian kernels and one versus all classifiers. Determine the weight of the SVM penalty and the width of the Gaussian Kernel using 10 fold cross validation. You are welcome to try to solve the problem with both standardized and raw attributes and report the results.

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

b2_best_c = []
b2_best_gamma = []
b2_best_c_std = []
b2_best_gamma_std = []

c_range = np.logspace(-2, 10, 5)
gamma_range = np.logspace(-9, 3, 5)

for col in y_train.columns: # calculate best c and gamma for each label separately
    b2_c_gamma = []
    b2_acc = []
    b2_acc_std = []
    for c in c_range:
        for gamma in gamma_range:
            b2_svc = SVC(C=c, gamma=gamma) # in SVC, default kernel is rbf; default decision_function_shape is ovr
            crossval_scores = cross_val_score(b2_svc,X_train,y_train[col],cv=10) # using raw data
            crossval_scores_std = cross_val_score(b2_svc,X_train_std,y_train[col],cv=10) # using standardized data
            b2_acc.append(crossval_scores.mean())
            b2_acc_std.append(crossval_scores_std.mean())
            b2_c_gamma.append((c, gamma))
    b2_best_c.append(b2_c_gamma[b2_acc.index(max(b2_acc))][0])
    b2_best_gamma.append(b2_c_gamma[b2_acc.index(max(b2_acc))][1])
    b2_best_c_std.append(b2_c_gamma[b2_acc_std.index(max(b2_acc_std))][0])
    b2_best_gamma_std.append(b2_c_gamma[b2_acc_std.index(max(b2_acc_std))][1])

print(y_train.columns)
print("Best C values",b2_best_c)
print("Best gamma values",b2_best_gamma)
print("Best C values for standardized data",b2_best_c_std)
print("Best gamma values for standardized data",b2_best_gamma_std)

# calculate exact match and hamming loss for each label
b2_exactmatch = []
b2_exactmatch_std = []
b2_hammingloss = []
b2_hammingloss_std = []
b2_y_pred = []
b2_y_pred_std = []
for i in range(len(y_train.columns)):
    b2_svc = SVC(C=b2_best_c[i], gamma=b2_best_gamma[i])
    b2_svc.fit(X_train,y_train[y_train.columns[i]])
    b2_pred = b2_svc.predict(X_test)
    b2_exactmatch.append(accuracy_score(y_test[y_test.columns[i]], b2_pred))
    b2_y_pred.append(b2_pred)
    b2_hammingloss.append(hamming_loss(y_test[y_test.columns[i]], b2_pred))
    # for standardized features
    b2_svc_std = SVC(C=b2_best_c_std[i], gamma=b2_best_gamma_std[i])
    b2_svc_std.fit(X_train_std,y_train[y_train.columns[i]])
    b2_pred_std = b2_svc_std.predict(X_test_std)
    b2_exactmatch_std.append(accuracy_score(y_test[y_test.columns[i]], b2_pred_std))
    b2_y_pred_std.append(b2_pred_std)
    b2_hammingloss_std.append(hamming_loss(y_test[y_test.columns[i]], b2_pred_std))

b2_y_pred = list(map(list,zip(*b2_y_pred)))
b2_y_pred_df = pd.DataFrame(b2_y_pred, columns=['Family','Genus','Species'])
b2_y_pred_df

b2_y_pred_std = list(map(list,zip(*b2_y_pred_std)))
b2_y_pred_std_df = pd.DataFrame(b2_y_pred_std, columns=['Family','Genus','Species'])
b2_y_pred_std_df

tbl = []
label = []
emr = []
ems = []
hlr = []
hls = []

for i in range(len(y_train.columns)):
    label.append(y_train.columns[i])
    emr.append(b2_exactmatch[i])
    ems.append(b2_exactmatch_std[i])
    hlr.append(b2_hammingloss[i])
    hls.append(b2_hammingloss_std[i])
tbl.append(label)
tbl.append(emr)
tbl.append(ems)
tbl.append(hlr)
tbl.append(hls)

tbl=list(map(list,zip(*tbl)))
tbl=pd.DataFrame(tbl, columns=['Label', 'Exact match(raw)', 'Exact match(standardized)', 'Hamming Loss(raw)', 'Hamming Loss(standardized)'])
tbl

b2_em = np.mean(np.all(np.equal(y_test,b2_y_pred_df), axis = 1))
print("Exact match is",b2_em)
b2_em_std = np.mean(np.all(np.equal(y_test,b2_y_pred_std_df), axis = 1))
print("Exact match(standardized features)is",b2_em_std)

b2_avg_hl = np.average(b2_hammingloss)
b2_avg_hl_std = np.average(b2_hammingloss_std)
print("Hamming loss",b2_avg_hl)
print("Hamming loss(standardized features)",b2_avg_hl_std)

# 1(b) iii. Repeat 1(b)ii with L1-penalized SVMs. Remember to standardize the attributes. Determine the weight of the SVM penalty using 10 fold cross validation.

b3_best_c_std = []

c_range = np.logspace(-2, 10, 5)

for col in y_train.columns:
    b3_acc_std = []
    b3_c=[]
    for c in c_range:
        b3_lsvc = LinearSVC(penalty='l1', C=c, dual=False)
        crossval_scores_std = cross_val_score(b3_lsvc,X_train_std,y_train[col],cv=10) # using standardized data
        b3_acc_std.append(crossval_scores_std.mean())
        b3_c.append(c)
    b3_best_c_std.append(b3_c[b3_acc_std.index(max(b3_acc_std))])

print(y_train.columns)
print("Best C values",b3_best_c_std)

# calculate exact match and hamming loss for each label
b3_exactmatch_std = []
b3_hammingloss_std = []
b3_y_pred = []

for i in range(len(y_train.columns)):
    b3_lsvc = LinearSVC(penalty='l1', C=b3_best_c_std[i], dual=False)
    b3_lsvc.fit(X_train_std,y_train[y_train.columns[i]])
    b3_pred = b3_lsvc.predict(X_test_std)
    b3_exactmatch_std.append(accuracy_score(y_test[y_test.columns[i]], b3_pred))
    b3_y_pred.append(b3_pred)
    b3_hammingloss_std.append(hamming_loss(y_test[y_test.columns[i]], b3_pred))

b3_y_pred = list(map(list,zip(*b3_y_pred)))
b3_y_pred_df = pd.DataFrame(b3_y_pred, columns=['Family','Genus','Species'])
b3_y_pred_df

tbl = []
label = []
ems = []
hls = []

for i in range(len(y_train.columns)):
    label.append(y_train.columns[i])
    ems.append(b3_exactmatch_std[i])
    hls.append(b3_hammingloss_std[i])
tbl.append(label)
tbl.append(ems)
tbl.append(hls)

tbl=list(map(list,zip(*tbl)))
tbl=pd.DataFrame(tbl, columns=['Label', 'Exact match(standardized)', 'Hamming Loss(standardized)'])
tbl

b3_em = np.mean(np.all(np.equal(y_test,b3_y_pred_df), axis = 1))
b3_avg_hl_std = np.average(b3_hammingloss_std)
print("Exact match(standardized features)",b3_em)
print("Hamming loss(standardized features)",b3_avg_hl_std)

# 1(b) iv. Repeat 1(b)iii by using SMOTE or any other method you know to remedy class imbalance. Report your conclusions about the classifiers you trained.

b4_best_c = []
b4_best_c_std = []

c_range = np.logspace(-2, 10, 5)

for col in y_train.columns:
    b4_acc_std = []
    b4_c=[]
    for c in c_range:
        b4_lsvc = LinearSVC(penalty='l1', C=c, dual=False, class_weight='balanced')
        crossval_scores_std = cross_val_score(b4_lsvc,X_train_std,y_train[col],cv=10) # using standardized data
        b4_acc_std.append(crossval_scores_std.mean())
        b4_c.append(c)
    b4_best_c_std.append(b4_c[b4_acc_std.index(max(b4_acc_std))])

print(y_train.columns)
print("Best C values",b4_best_c_std)

# calculate exact match and hamming loss for each label
b4_exactmatch_std = []
b4_hammingloss_std = []
b4_y_pred = []

for i in range(len(y_train.columns)):
    b4_lsvc = LinearSVC(penalty='l1', C=b4_best_c_std[i], dual=False, class_weight='balanced')
    b4_lsvc.fit(X_train_std,y_train[y_train.columns[i]])
    b4_pred = b4_lsvc.predict(X_test_std)
    b4_exactmatch_std.append(accuracy_score(y_test[y_test.columns[i]], b4_pred))
    b4_y_pred.append(b4_pred)
    b4_hammingloss_std.append(hamming_loss(y_test[y_test.columns[i]], b4_pred))

b4_y_pred = list(map(list,zip(*b4_y_pred)))
b4_y_pred_df = pd.DataFrame(b4_y_pred, columns=['Family','Genus','Species'])
b4_y_pred_df

tbl = []
label = []
ems = []
hls = []

for i in range(len(y_train.columns)):
    label.append(y_train.columns[i])
    ems.append(b4_exactmatch_std[i])
    hls.append(b4_hammingloss_std[i])
tbl.append(label)
tbl.append(ems)
tbl.append(hls)

tbl=list(map(list,zip(*tbl)))
tbl=pd.DataFrame(tbl, columns=['Label', 'Exact match(standardized)', 'Hamming Loss(standardized)'])
tbl

b4_em = np.mean(np.all(np.equal(y_test,b4_y_pred_df), axis = 1))
b4_avg_hl_std = np.average(b4_hammingloss_std)
print("Exact match(standardized features)",b4_em)
print("Hamming loss(standardized features)",b4_avg_hl_std)

# Highest exact match is obtained for SVM with Gaussian kernel and OVR classifier which is 0.9861046780917091.

# 2. K-Means Clustering on a Multi-Class and Multi-Label Data Set

import datetime
print(datetime.datetime.now())
best_k = []
hamming_distances = []
for m in range(50):
    print("Monte-Carlo iteration #",m+1)
    # (a) Choose k automatically based on one of the methods- CH or Gap Statistics or scree plots or Silhouettes
    sil_score = []
    for k in range(2, 51):
        kmeans = KMeans(n_clusters=k, random_state=np.random.randint(1000)).fit_predict(X)
        sil_score.append(silhouette_score(X, kmeans))
    best_k_val = 2+sil_score.index(max(sil_score))

    print("The best value of k is", best_k_val)
    kmeans = KMeans(n_clusters=best_k_val, random_state=np.random.randint(1000)).fit_predict(X)

    # (b) In each cluster, determine which family is the majority by reading the true labels. Repeat for genus and species.
    y_copy = y.copy()
    y_copy['cluster'] = kmeans
    fgs = []
    for cluster in range(best_k_val):
        family = y_copy.loc[y_copy['cluster']==cluster]['Family'].to_list()
        genus = y_copy.loc[y_copy['cluster']==cluster]['Genus'].to_list()
        species = y_copy.loc[y_copy['cluster']==cluster]['Species'].to_list()
        fgs.append([max(collections.Counter(family), key=collections.Counter(family).get),max(collections.Counter(genus), key=collections.Counter(genus).get),max(collections.Counter(species), key=collections.Counter(species).get)])
    print("Majority label triplet for each cluster is\n",fgs)

    # (c) Calculate the average Hamming distance, Hamming score, and Hamming loss between the true labels and the labels assigned by clusters.
    y_pred = []
    for i in y_copy['cluster']:
        y_pred.append(fgs[i])
    hammingdistance = np.sum(np.sum(np.not_equal(y, y_pred), axis=1))/float(y.shape[0])
#     avg_hammingdistance = np.average(hammingdistance)
    hammingloss = np.average(np.not_equal(y, y_pred))
    hammingscore = np.average(np.equal(y, y_pred))
#     avg_hammingscore = np.average(hammingscore)
#     avg_hammingloss = np.average(hammingloss)
    hamming_distances.append(hammingdistance)
    print("Hamming distance is\n", hammingdistance)
    print("Hamming score is\n", hammingscore)
    print("Hamming loss is\n", hammingloss)

    print("---------------------------------------------------")

hd_avg_arr = np.array(hamming_distances)
hd_avg = hd_avg_arr.mean()
print("Average Hamming Distance over 50 Monte-Carlo iterations is",hd_avg)
hd_std = hd_avg_arr.std()
print("Standard Deviation of 50 Hamming Distances is",hd_std)
print(datetime.datetime.now())
