# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np 
import pandas as pd 
import cv2




# 1. receba como argumento o diretório de imagens de números
if (len(sys.argv) < 3):
    print("Passe o caminho das imagens e o caminho do arquivo files.txt")
    print("USAGE: python2.7 feature_extraction.py digits2k/data file.txt")
    exit(1)

    
image_dir = sys.argv[1]
labels = sys.argv[2]

fp = sorted(open(labels).readlines())
ff = [x.split('/')[1] for x in fp]

with open("imagens_rotuladas.csv", "w") as ir:
    # 2. transforme cada imagem em um vetor de características/atributos/pixels
    for img_lbl in zip(sorted(os.listdir(image_dir)), ff):   
        img, lbl = img_lbl[1].split()
        print("Image: {} \t Label: {}".format(img, lbl))
        im = cv2.imread(os.path.join(image_dir, img))
        
        # 3. adicione um rótulo ao vetor, correlacionando os números com o nome do arquivo da imagem (files.txt)
        # 4. escreva os vetores que representam as imagens rotuladas em um arquivo, onde cada vetor é uma linha desse arquivo
        ir.write("{},{},{}".format(
           img,
           ','.join(str(s) for s in im.flatten()),
           lbl
        ))

