#!/usr/bin/python

#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file, make_classification, make_gaussian_quantiles, make_blobs
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



def GridSearch(X_train, y_train):

	# define range dos parametros
	C_range = 2. ** numpy.arange(-5,15,2)
	gamma_range = 2. ** numpy.arange(3,-15,-2)
	#k = [ 'rbf']
	k = ['linear', 'rbf']
	param_grid = dict(gamma=gamma_range, C=C_range, kernel=k)

	# instancia o classificador, gerando probabilidades
	srv = svm.SVC(probability=True)

	# faz a busca
	grid = GridSearchCV(srv, param_grid, n_jobs=-2, verbose=True)
	grid.fit (X_train, y_train)

	# recupera o melhor modelo
	model = grid.best_estimator_
	
	# imprime os parametros desse modelo
	print(grid.best_params_)
	return model
	
	
	
	
	
def main():

	## create data...
	plt.figure(figsize=(8, 8))	
	
	print("'kernel', 'round', 'score', 'vetores'")
	for kernel in ['linear', 'rbf']:
		for i in range(10):
	
			#X, y = make_blobs(n_samples=300, centers=2)
			
			X, y = make_gaussian_quantiles(n_samples =300, n_features=2, n_classes =2)

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
		
			# cria um SVM
			clf =  svm.SVC(kernel=kernel)

			# treina o classificador na base de treinamento
			#print "Training Classifier..."
			clf.fit(X_train, y_train)
	
			print("[\'{}\',{},{},{}], ".format(
					kernel, i, 
					clf.score(X_test, y_test), 
					clf.n_support_[0]+clf.n_support_[1]))
			

	

	# # GridSearch retorna o melhor modelo encontrado na busca
	# best = GridSearch(X_train, y_train)

	# # resultado do treinamento
	# print ('Accuracia no kernl RBF:', )
	# print best.score(X_train, y_train)
	# print ('Vetores de suporte:',  best.n_support_[0]+best.n_support_[1] )


	# Treina usando o melhor modelo
	# best.fit(X_train, y_train)
#

	# # resultado do treinamento
	# print ('Accuracia no teste:', )
	# print best.score(X_test, y_test)

	# # predicao do classificador
	# y_pred = best.predict(X_test)

	# # cria a matriz de confusao
	# cm = confusion_matrix(y_test, y_pred)
	# print cm



if __name__ == "__main__":
	if len(sys.argv) != 1:
		sys.exit("Use: svmtoy.py")

	main()