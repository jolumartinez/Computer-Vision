# -*- coding: utf-8 -*-
"""
Created on Wed Apr 8 22:22:30 2020

@author: jolumartinez
"""

import cv2
from mis_clases import proyectividad as prm

images_alignment = prm.ProyectividadOpenCV()

#It defines image size
width, height = 700, 500

nombre_carpeta = "sequoia_images"

#Reading images
img_RGB = cv2.imread(nombre_carpeta + "img_RGB.JPG",0)
img_GRE = cv2.imread(nombre_carpeta + "img_GRE.TIF",0)
img_NIR = cv2.imread(nombre_carpeta + "img_NIR.TIF",0)
img_RED = cv2.imread(nombre_carpeta + "img_RED.TIF",0)
img_REG = cv2.imread(nombre_carpeta + "img_REG.TIF",0)

merged_fix_bad = cv2.merge((img_GRE, img_RED, img_NIR))

stb_RGB, stb_GRE, stb_NIR, stb_RED, stb_REG = images_alignment.img_alignment_sequoia(img_RGB, img_GRE, img_NIR, img_RED, img_REG, width, height)

merged_fix_stb = cv2.merge((stb_GRE, stb_RED, stb_NIR))

cv2.imshow('frame', merged_fix_bad)
cv2.waitKey(0)

cv2.imshow('frame', merged_fix_stb)
cv2.waitKey(0)

cv2.destroyAllWindows()
