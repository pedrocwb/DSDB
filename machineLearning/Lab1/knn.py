import os
import sys
import cv2
import random
import numpy as np
import pylab as pl
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

print(cv2.__version__)


def load_images(path_images, X, Y, fout):
    #print ('Loading images...')
    archives = os.listdir(path_images)
    images = []
    arq = open('digits/files.txt')
    lines = arq.readlines()
    
    for line in lines:
        aux = line.split('/')[1]
        image_name = aux.split(' ')[0]
        label = line.split(' ')[1]
        label = label.split('\n')
        for archive in archives:
            if archive == image_name:
                path  = path_images + archive
                image = Image.open(path).convert('RGB') 
                image = np.array(image)[:, :, ::-1].copy()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rawpixel(image, label[0], X, Y, fout)


#########################################################
# Usa o valor dos pixels como caracteristica
#
#########################################################


def rawpixel(image, label, X, Y, fout):
    
    image = cv2.resize(image, (X, Y), interpolation=cv2.INTER_CUBIC)
    fout.write(str(label) + " ")

    indice = 0
    for i in range(Y):
        #vet= []
        for j in range(X):
            if(image[i][j] > 250):
                v = 0
            else:
                v = 1
            # vet.append(v)

            fout.write(str(indice)+":"+str(v)+" ")
            indice = indice+1

    fout.write("\n")


def main(data, k=5, metric='euclidean'):

    # loads data
    X_data, y_data = load_svmlight_file(data)
    # splits data
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.5, random_state=111)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    def addFeat(feat):
        return [np.mean(feat), np.average(feat), np.var(feat), np.std(feat)]

    X_train = [np.append(feat, addFeat(feat)) for feat in X_train]
    X_test = [np.append(feat, addFeat(feat)) for feat in X_test]

    # fazer a normalizacao dos dados #######
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # cria um kNN
    neigh = KNeighborsClassifier(n_neighbors=k, metric=metric)

    neigh.fit(X_train, y_train)

    # predicao do classificador
    y_pred = neigh.predict(X_test)

    # mostra o resultado do classificador na base de teste
    score = neigh.score(X_test, y_test)
    print(score)

    # cria a matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    # pl.matshow(cm)
    # pl.colorbar()
    # pl.show()
    return (score, cm)


def extractRepresentation(file, X, Y):
    fout = open(file, "w")
    load_images('digits/data/', X, Y, fout)
    fout.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Use: knn.py <data>")

    results = []
    for x in range(25, 30, 5):

        X, Y = x, x
        print("Image shape {}x{} Score: ".format(X, Y), end = " ")
        extractRepresentation(sys.argv[1], X, Y)
        score, cm = main(sys.argv[1])
        results.append((score, (X, Y), cm))



    from operator import itemgetter
    
    worst = min(results, key=itemgetter(0))
    optimal = max(results, key=itemgetter(0))

    print("Optimal shape {} score = {}".format(optimal[1], optimal[0]))
    print("Worst shape {} score = {}".format(worst[1], worst[0]))
    
    labelsint = range(0, len(results))
    labels = [ "{}x{}".format(i[1][0], i[1][0])  for i in results ]
    values = [ i[0] for i in results ]

    fig, ax = plt.subplots()
    plt.plot(labelsint, values)
    plt.xticks(labelsint, labels)
    plt.title("Shape x Score")
    plt.xlabel("Image shape")
    plt.ylabel("Precision Score")
    plt.savefig("shape_score.png")
    
    print(worst[2])
    pl.matshow(worst[2])
    pl.title("worst")
    pl.colorbar()
    pl.savefig("shape_worst.png")

    print(optimal[2])
    pl.matshow(optimal[2])
    pl.title("optimal")
    pl.colorbar()
    pl.savefig("shape_optimal.png")
    

    X_optimal, Y_optimal = optimal[1]
    print("==" * 10)
    print("Find optimal K and Metric: {}x{}".format(X_optimal, Y_optimal))

    extractRepresentation(sys.argv[1], X_optimal, Y_optimal)
    results = []
    for metric in ["euclidean", "manhattan", "chebyshev", "mahalanobis"]:
        results2 = []
        for k in range(1, 16, 1):
            print("Metric: {} K = {} score = ".format(
                metric, k), end=" ")
            score, cm = main(sys.argv[1], k, metric)
            results.append((score, (X_optimal, Y_optimal), k, metric, cm))
            results2.append((score, (X_optimal, Y_optimal), k, metric, cm))

            
        X = [ i[2] for i in results2 ]
        values = [ i[0] for i in results2 ]

        fig, ax = plt.subplots()
        plt.plot(X, values)
        plt.xlabel("Neighbors")
        plt.ylabel("Precision Score")
        plt.title(metric)
        plt.savefig("{}_score.png".format(metric))
        

    worst = min(results, key=itemgetter(0))
    optimal = max(results, key=itemgetter(0))

    print("Optimal: {} score = {} K = {} metric = {}".format(
        optimal[1], optimal[0], optimal[2], optimal[3]))
    print("Worst: {} score = {} K = {} metric = {}".format(
        worst[1], worst[0], worst[2], worst[3]))

    print(worst[4])
    pl.matshow(worst[4])
    pl.title("worst")
    pl.colorbar()
    pl.savefig("worst_metric.png")

    print(optimal[4])
    pl.matshow(optimal[4])
    pl.title("optimal")
    pl.colorbar()
    pl.savefig("optimal_metric.png")
    