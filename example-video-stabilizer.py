# -*- coding: utf-8 -*-
"""
Created on Fri March 14:1:30 2020

@author: jolumartinez
"""

import cv2
from visionPC import projectivity as prm

cap = cv2.VideoCapture('billarVideo.mp4')

img_base = cv2.imread("baseBillar.png")
img_base = cv2.resize(img_base, (900, 500), interpolation = cv2.INTER_LINEAR)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (900,500))

estabilizador = prm.ProyectividadOpenCV()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (900, 500), interpolation = cv2.INTER_LINEAR)
    estabilizada = estabilizador.estabilizador_imagen(frame,img_base)
    out.write(estabilizada)

    # Display the resulting frame
    cv2.imshow('frame',estabilizada)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()