# coding: utf-8
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix 
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
X_train, y_train = load_svmlight_file('train.txt')
X_test, y_test = load_svmlight_file('test.txt')
size = X_train.shape
print("Training size", size)
print("Testing size", X_test.shape)
### Definição da função fit que vai treinar, executar a função predict e retornar a matriz de confusão e score
Determinando o tamanho dos batchs
de 100 até 1000 os tamanhos variam a um passo de 100, de 1000 até o tamanho máximo da base o batch varia a um passo de 1000
batchs = list(range(100, 1001, 100)) + list(range(1000, size[0] + 1, 1000))
batchs = list(range(100, 1001, 100)) + list(range(1000, size[0] + 1, 1000))
print(batchs)
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

history = []
for clf in classifiers:
    for batch in batchs:
        

        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        

        print("{} - Training size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
def fit(X_train, y_train, X_test, y_test, clf):

    X_train_dense = X_train.toarray()
    clf.fit(X_train_dense, y_train)

    X_test_dense = X_test.toarray()
    y_pred = clf.predict(X_test_dense) 
    
    cm = confusion_matrix(y_test, y_pred)
    
    # mostra o resultado do classificador na base de teste
    return clf.score(X_test_dense, y_test), cm
batchs = list(range(100, 1001, 100)) + list(range(1000, size[0] + 1, 1000))
print(batchs)
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

history = []
for clf in classifiers:
    for batch in batchs:
        

        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        

        print("{} - Training size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

history = []
for clf in classifiers:
    for batch in batchs:
        

        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        

        print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

history = []
for clf in classifiers:
    for ix, batch in enumerate(batchs):
        

        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        
        if ix % 10 == 0:
            print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

history = []
for clf in classifiers:
    for ix, batch in enumerate(batchs):
        

        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        
        if ix % 2 == 0:
            print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

history = []
for clf in classifiers:
    for ix, batch in enumerate(batchs):
        

        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        
        if ix % 12 == 0:
            print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

history = []
for clf in classifiers:
    for ix, batch in enumerate(batchs):
        

        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        
        if ix % 20 == 0:
            print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

history = []
for clf in classifiers:
    for ix, batch in enumerate(batchs):
        

        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        
        
        print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

history = []
for clf in classifiers:
    for ix, batch in enumerate(batchs):
        

        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        
        if ix % 10 == 2:
            print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

history = []
for clf in classifiers:
    for ix, batch in enumerate(batchs):
        
        
        print(batchs)
        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        
        if ix % 10 == 2:
            print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

history = []
for clf in classifiers:
    for ix, batch in enumerate(batchs):
        
        
        print(batch)
        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        
        if ix % 10 == 2:
            print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

history = []
for clf in classifiers:
    for ix, batch in enumerate(batchs):
                
        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        
        if ix % 11 == 2:
            print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
batchs = list(range(100, 1001, 100)) + list(range(1000, size[0] + 1, 1000))
print(len(batchs))
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}
len(batchs)
history = []
for clf in classifiers:
    for ix, batch in enumerate(batchs):
                
        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        
        
        if ix + 1 % 10 == 0:
            print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
batchs = list(range(100, 1001, 100)) + list(range(1000, size[0] + 1, 1000))
print(batchs)
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}
history = []
for clf in classifiers:
    for ix, batch in enumerate(batchs):
                
        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        
        
        if ix + 1 % 10 == 0:
            print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
classifiers = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "LDA" : LinearDiscriminantAnalysis(),
    "Naive Bayes" : GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}
history = []
for clf in classifiers:
    for ix, batch in enumerate(batchs):
                
        xt = X_train[0:batch]
        yt = y_train[0:batch]

        tic = time.time()
        score, cm = fit(xt, yt, X_test, y_test, classifiers[clf])
        toc = time.time()
        
        
        if (ix + 1) % 10 == 0:
            print("{} - Batch size: {} - t: {}".format(clf, batch, toc - tic))
        
        history.append([clf, batch, score, toc-tic, cm])

    results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time', 'confusion_matrix'])
sns.lineplot(x = 'batch', y='score', hue="Classifiers", data=results)
sns.lineplot(x = 'batch', y='score', style= "Classifiers", hue="Classifiers", data=results)
sns.lineplot(x = 'batch', y='time', style= "Classifiers", hue="Classifiers", data=results)
sns.lineplot(x = 'batch', y='time', style= "Classifiers", hue="Classifiers", data=results[results["Classifiers"] != "KNN"])
sns.lineplot(x = 'batch', y='time', style= "Classifiers", hue="Classifiers", data=results[results["Classifiers"] == "KNN"])
ax = sns.lineplot(x = 'batch', y='time', style= "Classifiers", hue="Classifiers", data=results[results["Classifiers"] != "KNN"])
ax = sns.lineplot(x = 'batch', y='time', style= "Classifiers", hue="Classifiers", data=results[results["Classifiers"] != "KNN"])
ax.set(ylabel=" Time (s)")
ax = sns.lineplot(x = 'batch', y='time', style= "Classifiers", hue="Classifiers", data=results[results["Classifiers"] == "KNN"])
ax.set(ylabel=" Time (s)")
300*60
300/60
(300 * 60 ) * 60
(300 / 60 )
(300 / 60 ) * 60
results[results["Classifiers"] == "KNN"]
results[results["Classifiers"] == "KNN"].groupby(["KNN"])["time"].cumsum()
results[results["Classifiers"] == "KNN"].groupby(["KNN"])["time"].sum()
results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].cumsum()
knntime = results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
knntime = results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
knntime
knntime = results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
print(knntime, ' s')
knntime = results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
print(knntime["KNN"], ' s')
knntime = results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
print(knntime["KNN"], 'segundos')
knntime = results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
knntime = knntime["KNN"]
knntime = results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
knntime = knntime["KNN"]
knntime
knntime = results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
knntime = knntime["KNN"]
knntime/60 
knntime = results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
knntime = knntime["KNN"]
(knntime/60) /12 
knntime = results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
knntime = knntime["KNN"]
(knntime/60) / 60
knntime = results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
knntime = knntime["KNN"]
knntime/60
knntime = results[results["Classifiers"] == "KNN"].groupby(["Classifiers"])["time"].sum()
knntime = knntime["KNN"]
print("tempo total KNN: {} minutos".format(knntime/60))
ax = sns.lineplot(x = 'batch', y='score', style= "Classifiers", hue="Classifiers", data=results, figsize=(10,5))
fig, ax = plt.subplots(figsize(10,5))
ax = sns.lineplot(x = 'batch', y='score', style= "Classifiers", hue="Classifiers", data=results)
fig, ax = plt.subplots(figsize=(10,5))
ax = sns.lineplot(x = 'batch', y='score', style= "Classifiers", hue="Classifiers", data=results)
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.lineplot(x = 'batch', y='score', style= "Classifiers", hue="Classifiers", data=results)
fig, ax = plt.subplots(figsize=(10,7))
ax = sns.lineplot(x = 'batch', y='score', style= "Classifiers", hue="Classifiers", data=results)
fig, ax = plt.subplots(figsize=(10,7))
ax = sns.lineplot(x = 'batch', y='score', style= "Classifiers", hue="Classifiers", data=results)
plt.axvline(10000)
gt_10000 = resultados[resultados["score"] >= 10000]
gt_10000 = results[results["score"] >= 10000]
gt_10000 = results[results["score"] >= 10000]
gt_10000
gt_10000 = results[results["batcg"] >= 10000]
gt_10000
gt_10000 = results[results["batch"] >= 10000]
gt_10000
gt_10000 = results[results["batch"] >= 10000]
gt_10000.groupby(["Classifiers"])["score"].var()
gt_10000 = results[results["batch"] >= 10000]
gt_10000.groupby(["Classifiers"])["score"].std()
gt_10000 = results[results["batch"] <= 10000]
gt_10000.groupby(["Classifiers"])["score"].std()
gt_10000 = results[results["batch"] >= 10000]
gt_10000.groupby(["Classifiers"])["score"].std()
gt_10000 = results[results["batch"]]
gt_10000 = results[results["batch"] >= 10000, "score"]
gt_10000 = results["score", results["batch"] >= 10000]
gt_10000 = results[results["batch"] >= 10000][""score""]
gt_10000 = results[results["batch"] >= 10000]["score"]
gt_10000 = results[results["batch"] >= 10000]["score"]
gt_10000
[results[results["batch"] <= batch]["score"], "score"] for batch in batchs]
[results[results["batch"] <= batch]["score"] for batch in batchs]
gt_10000 = results[results["batch"] >= 10000]
list(gt_10000.groupby(["Classifiers"])["score"].std())
gt_10000 = results[results["batch"] >= 10000]
ist(gt_10000.groupby(["Classifiers"])["score"].std()
gt_10000 = results[results["batch"] >= 10000]
gt_10000.groupby(["Classifiers"])["score"].std()
gt_10000 = results[results["batch"] >= 10000]
pd.DataFrame(gt_10000.groupby(["Classifiers"])["score"].std())
gt_10000 = results[results["batch"] >= 10000]
pd.DataFrame(gt_10000.groupby(["Classifiers"])["score"].std(), columns = ["Classifiers", "Desvio Padrão"])
gt_10000 = results[results["batch"] >= 10000]
pd.DataFrame(gt_10000.groupby(["Classifiers"])["score"].std())
gt_10000 = results[results["batch"] <= 10000]
pd.DataFrame(gt_10000.groupby(["Classifiers"])["score"].std())
gt_10000 = results[results["batch"] > 10000]
pd.DataFrame(gt_10000.groupby(["Classifiers"])["score"].std())
gt_10000 = results[results["batch"] >= 10000]
pd.DataFrame(gt_10000.groupby(["Classifiers"])["score"].std())
tst = results[results["batch"] >= 10000]
pd.DataFrame(tst.groupby(["Classifiers"])["score"].std())
tst = results[results["batch"] >= 10000]
print(pd.DataFrame(tst.groupby(["Classifiers"])["score"].std()))
tst = results[results["batch"] >= 10000]
print(tst.groupby(["Classifiers"])["score"].std())
tst = results[results["batch"] < 10000]
print(tst.groupby(["Classifiers"])["score"].std())
tst = results[results["batch"] < 10000]
print(tst.groupby(["Classifiers"])["score"].std())
tst = results[results["batch"] >= 10000]
pd.DataFrame(tst.groupby(["Classifiers"])["score"].std())
tst = results[results["batch"] < 10000]
print(tst.groupby(["Classifiers"])["score"].std())
dt = results[results["batchs"] < 1000]
dt = results[results["batch"] < 1000]
dt = results[results["batch"] < 1000]
dt = results[results["batch"] < 1000]
dt.groupby("Classifier")["score"].max()
dt = results[results["batch"] < 1000]
dt.groupby("Classifiers")["score"].max()
dt.groupby("Classifiers")["time"].max()
dt.groupby("Classifiers")["time"].min()
dt = results[results["batch"] < 1000]
fig, ax = plt.subplots(figsize=(10,7))
ax = sns.lineplot(x = 'batch', y='score', style= "Classifiers", hue="Classifiers", data=dt)
dt.groupby("Classifiers")["score"].max()
dt.groupby("Classifiers")["time"].min()
dt.groupby("Classifiers")["time"].min()
ax = sns.lineplot(x = 'batch', y='time', style= "Classifiers", hue="Classifiers", data=dt[dt["Classifiers"] != "KNN"])
ax.set(ylabel=" Time (s)")
get_ipython().run_line_magic('save', 'mysession')
get_ipython().run_line_magic('save', '')
get_ipython().run_line_magic('save', '"mysession"')
get_ipython().run_line_magic('pinfo', '%save')
get_ipython().run_line_magic('save', '%save -r mysession 1-99999')
