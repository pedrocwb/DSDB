import sys
import numpy as np
import time

from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import confusion_matrix 
from sklearn.datasets import load_svmlight_file

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main(X_train, y_train, X_test, y_test, clf):

	X_train_dense = X_train.toarray()
	clf.fit(X_train_dense, y_train)

	X_test_dense = X_test.toarray()
	y_pred = clf.predict(X_test_dense) 
	
	# mostra o resultado do classificador na base de teste
	return clf.score(X_test_dense, y_test)

	# cria a matriz de confusao
	#cm = confusion_matrix(y_test, y_pred)
	#print cm

	#print y_predProb



if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Use: lda.py <dataTR> <dataTS>")


#	print(list(range(0, 1001, 100)) + list(range(1000, 20000, 1000)))
	# loads data
	print ("Loading data...")
	X_train, y_train = load_svmlight_file(sys.argv[1])
	X_test, y_test = load_svmlight_file(sys.argv[2])
	size = X_train.shape

	batchs = list(range(100, 1001, 100)) + list(range(1000, size[0] + 1, 1000))
	

	classifiers = {
		#"Logistic Regression": linear_model.LogisticRegression(),
		#"Naive Bayes" : GaussianNB(),
		#"LDA" : LinearDiscriminantAnalysis()
		"KNN": KNeighborsClassifier(n_neighbors=3, metric='euclidean')
	}
	history = []
	for clf in classifiers:	
		for batch in batchs[:4]:
			print("Training size: %d" % batch, end = ' ')

			xt = X_train[0:batch]
			yt = y_train[0:batch]

			tic = time.time()
			score = main(xt, yt, X_test, y_test, classifiers[clf])
			toc = time.time()
			print(toc - tic)

			history.append([clf, batch, score, toc-tic])

		results = pd.DataFrame(history, columns=['Classifiers', 'batch', 'score', 'time'])
		
		
		# fig, ax1 = plt.subplots(1,1)
		# ax1 = sns.lineplot(x = 'batch', y='score', data=results)
		# ax1.set(title=clf)

		# fig, ax2 = plt.subplots(1,1)
		# ax2 = sns.lineplot(x = 'batch', y='time', data=results)
		# ax2.set(ylabel="Time (s)", title=clf)
		
		# plt.show()
	
	
	less_than_thousand = results[results["batch"] < 1000]

	print(less_than_thousand.groupby(["Classifiers"])["score"].max())

	

