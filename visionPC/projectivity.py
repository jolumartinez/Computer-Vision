#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method collection for computer vision

Created on Wed Sep 13 18:07:14 2017
@author: jolumartinez
"""

# Imports

import cv2
import numpy as np
import os.path
#import serial
import math
from scipy.interpolate import interp1d
from time import time


puntos_click = list()
# ================================================================
def click_and_count(event, x, y, flags, param):
    """Definicion del callback que atiende el mouse"""

    global puntos_click

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        puntos_click.append((x, y))

# +=================================================================================================
class ProyectividadOpenCV():
    """
    This class contains several methods for computer vision
    """
    # Atributos de la clase
    error_reproyeccion = 4
    #--------------------------------------------------------------------------    
    def __init__(self):
        """Inicializador del objeto miembro de la clase"""
        
    
    #--------------------------------------------------------------------------    
    def coser_imagenes(self, ruta_img_base, ruta_img_adicional, radio = 0.75, error_reproyeccion = 4.0, coincidencias = False):
        """Method for stitching two images"""
        
        imagen_adicional = ruta_img_adicional
        imagen_base = ruta_img_base
        # Se obtienen los puntos deinterés
        
        (kpsBase, featuresBase) = self.obtener_puntos_interes(imagen_base)
        (kpsAdicional, featuresAdicional) = self.obtener_puntos_interes(imagen_adicional)
        # Se buscan las coincidencias
        
        M = self.encontrar_coincidencias(imagen_base, imagen_adicional, kpsBase, kpsAdicional, featuresBase, featuresAdicional, radio)
        
        if M is None:
            return None
        
        # Se halla la homgrafia
        (H, status) = self.encontrar_H_RANSAC(M, kpsBase, kpsAdicional, error_reproyeccion)
              
        # Organizando la imagen resultante
        
        result = cv2.warpPerspective(imagen_base, H, (imagen_base.shape[1], imagen_base.shape[0]))
        result[0:imagen_adicional.shape[0], 0:imagen_adicional.shape[1]] = imagen_adicional

        # check to see if the keypoint matches should be visualized
        if coincidencias:
            vis = self.dibujar_coincidencias(imagen_base, imagen_adicional, kpsBase, kpsAdicional, M, status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result
    
    #--------------------------------------------------------------------------
    def estabilizador_imagen(self, imagen_base, imagen_a_estabilizar, radio = 0.75, error_reproyeccion = 4.0, coincidencias = False):
        """Esta clase devuelve una secuencia de imágenes tomadas de la cámara estabilizada con respecto a la primera imagen"""
        
        # Se obtienen los puntos deinterés
        
        (kpsBase, featuresBase) = self.obtener_puntos_interes(imagen_base)
        (kpsAdicional, featuresAdicional) = self.obtener_puntos_interes(imagen_a_estabilizar)
        # Se buscan las coincidencias        
        
        M = self.encontrar_coincidencias(imagen_base, imagen_a_estabilizar, kpsBase, kpsAdicional, featuresBase, featuresAdicional, radio)
        
        if M is None:
            print("pocas coincidencias")
            return None
        
        if len(M) > 4:
            # construct the two sets of points
            
#            M2 = cv2.getPerspectiveTransform(ptsA,ptsB)
            (H, status) = self.encontrar_H_RANSAC_Estable(M, kpsBase, kpsAdicional, error_reproyeccion)
            estabilizada = cv2.warpPerspective(imagen_base,H,(imagen_base.shape[1],imagen_base.shape[0]))
            return estabilizada
        print("sin coincidencias")
        return None
        
            
     
    #--------------------------------------------------------------------------
    def obtener_puntos_interes(self, imagen):
        """Se obtienen los puntos de interes cn SIFT"""
        
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(imagen, None)
        
        return kps, features
    
    #--------------------------------------------------------------------------
    def encontrar_coincidencias(self, img1, img2, kpsA, kpsB, featuresA, featuresB, ratio):
        """Metodo para estimar la homografia"""
        
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
#        
#        # loop over the raw matches
        for m in rawMatches:
#            # ensure the distance is within a certain ratio of each
#            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        
#        print (matches)
        return matches
    
    #--------------------------------------------------------------------------
    def encontrar_H_RANSAC(self, matches, kpsA, kpsB, reprojThresh):
        """Metodo para aplicar una H a una imagen y obtener la proyectividad"""

        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsB[i].pt for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (H, status)

        # otherwise, no homograpy could be computed
        return None
    
    
    #--------------------------------------------------------------------------
    def encontrar_H_RANSAC_Estable(self, matches, kpsA, kpsB, reprojThresh):
        """Metodo para aplicar una H a una imagen y obtener la proyectividad"""
        
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsB[i].pt for (i, _) in matches])
    
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            
            return (H, status)

        return None

    def dibujar_coincidencias(self, imagen_base, imagen_adicional, kpsA, kpsB, matches, status):
        
        (hA, wA) = imagen_base.shape[:2]
        (hB, wB) = imagen_adicional.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imagen_base
        vis[0:hB, wA:] = imagen_adicional

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
                ptB = (int(kpsB[trainIdx].pt[0]) + wA, int(kpsB[trainIdx].pt[1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis
    

    #--------------------------------------------------------------------------
    def encontrar_H_marcas(self, las_xprima, las_x):
        """Metodo para estimar la homografia"""
        #Se utiliza 0 y no RANSAC porque deseo que utilice todos los puntos que se tienen
        H, estado = cv2.findHomography(las_x, las_xprima, 0,0.1)
        return H, estado

    #--------------------------------------------------------------------------
    def estabilizar_desde_marcas(self, imagen, marcas_click, marcas_cad_mm):
        """Esta clase retorna una imagen estabilizada con base en una imagen abstraida delas marcas del cad dadas en mm"""
        
        #Lo primero es tratar la imagen entrante
        blur = cv2.blur(imagen, (3,3))
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        
        #Se aplica un filtro de color verde para separar las marcas del fondo
        thresh_marcas = cv2.inRange(hsv,np.array((49,50,50)), np.array((107, 255, 255)))
        marcas_H = list()
        cad_H = list()
        #Se hace una busqueda de las marcas visibles en un radio de 30 pixelesy se filtranpor area para sacar los pares que permitiran hallar la homografia
        for i in range(0,len(marcas_click)):
#            print(i)
            x_men=marcas_click[i][0]-10
            x_may=marcas_click[i][0]+10
            y_men=marcas_click[i][1]-10
            y_may=marcas_click[i][1]+10
            area_marca = thresh_marcas[y_men:y_may, x_men:x_may]
            image_marcas, contours_marcas,hierarchy_marcas = cv2.findContours(area_marca,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            
            max_area = 65
            best_cnt = 1
            for cnt in contours_marcas:
                area = 1
                area = cv2.contourArea(cnt)
#                print(contours_marcas)
#                print(area,m)
                if area > max_area and area < 85:
                    max_area = area
                    best_cnt = cnt
                    # finding centroids of best_cnt and draw a circle there
                    cM = cv2.moments(best_cnt)
                    cx,cy = int(cM['m10']/cM['m00']), int(cM['m01']/cM['m00'])
                    cx = x_men+cx
                    cy = y_men+cy
                    marcas_H.append((cx,cy))
                    cad_H.append((marcas_cad_mm[i][0]+100,marcas_cad_mm[i][1]))
#                    print(marcas_H,cad_H)
                    
                    
        las_x = np.array(cad_H)
        las_xprima = np.array(marcas_H)
        print(las_x,las_xprima)
        if len(las_xprima) > 4:
            
            H, estado = self.encontrar_H_marcas(las_x, las_xprima)

            estabilizada = cv2.warpPerspective(imagen,H,(1200,1200))
            
            return estabilizada
        
        return None
    
    #--------------------------------------------------------------------------
    def estabilizar_desde_centroides_marcas(self, imagen, marcas_click, marcas_cad_mm):
        """Esta clase retorna una imagen estabilizada con base en una imagen abstraida delas marcas del cad dadas en mm"""
        
        las_x = np.array(marcas_cad_mm)
        las_xprima = np.array(marcas_click)
        print(las_x,las_xprima)
        if len(las_xprima) > 4:
            
            H, estado = self.encontrar_H_marcas(las_x, las_xprima)

            estabilizada = cv2.warpPerspective(imagen,H,(650,650))
            
            return estabilizada
        
        return None
            
            
    #--------------------------------------------------------------------------
    def inicializar_marcas(self, img_base):
        """Permite al usuario hacer click en el centro apriximado de cada marca y las guarda en orden"""
        
        global puntos_click
        
        #Copiar la imagen original para poder escribir sobre ella
        #Sin modificarla
        imagen_conmarcas =self.img_base.copy()
        
        #Mostrar la imagen
        cv2.namedWindow("Imagen_base")
        cv2.setMouseCallback("Imagen_base", click_and_count)
        
        while True:
            # Mostrar a imagen
            cv2.imshow("Imagen_base", imagen_conmarcas)
            key = cv2.waitKey(1) & 0xFF
            
            # Menu principal
            #Si se presiona r resetee la imagen
            if key == ord("r"):
                puntos_click = list()
                imagen_conmarcas = self.img_base.copy()
                
            # Si se presiona q salir
            elif key == ord("q"):
                return puntos_click
                break
        
            # Graficar los puntos que hayan en puntos_click
            if puntos_click:
                for pts_id, coords in enumerate(puntos_click):
                    #Coordenadas
                    x, y = coords[0], coords[1]
                    # Dibujar un circulo
                    cv2.circle(imagen_conmarcas, (x, y), 5, (0,0,255), 5, 2)
                    # Seleccionar una fuente
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(imagen_conmarcas, str(pts_id+1), (x, y), font, 10, (0,0,255),5)
                    
    #------------------------------------------------------------------------------------------------
    
    def cinematica_inversa(self, a, b):
        """Recibe las coordenadas en centimetros"""
        #Se valida que la coordenada se encuentre dentro del area de trabajo. Si no esta se ubica en el centro el manipulador
        if a <= 14 or a >= 34:
            a = 15
        if b <= 4 or b >= 26:
            b = 16
            
        xp = a
        yp = b
            
            
        xa=0
        ya=0
        xb=46
        yb=0
        xc=46/2
        yc=46*math.sin(math.pi/3)
        
        # Dimenciones de los eslabones y acopladores del manipulador (Unidades en cm.) 
        manivela=19
        
        # Eslabones (Cadenas Cinematica 1)
        l1=manivela
        L_DA=l1
        l4=manivela
        L_GD=l4
        
        # Eslabones (Cadena Cinematica 2)
        l2=manivela
        L_EB=l2
        l5=manivela
        L_HE=l5
        
        # Eslabones (Cadena Cinematica 3)
        l3=manivela
        L_FC=l3
        l6=manivela
        L_IF=l6
        
        # plataforma movil
        # El parametro (h) representa el baricentro de la plataforma.
        h=7.5056
        #phi=0; % Plataforma no rota.
        phi=0*math.pi/180 # Plataforma rota.
        
        # Datos de conversión (Grados --> Bits)
        
        grados=360
        decimal=4096
        #    Resolucion del servomotor.
        
    #        xp=23#               23.5
    #        yp=13.279056191#       13.279056191
                  
        # Coordenadas de la plataforma movil 
        # Coordenadas del punto G.
        xg=xp-h*math.cos(phi+(math.pi/6))
        yg=yp-h*math.sin(phi+(math.pi/6))
        # Coordenadas del punto H.
        xh=xp-h*math.cos(phi+(math.pi-(math.pi/6)))
        yh=yp-h*math.sin(phi+(math.pi-(math.pi/6)))
            
        # Coodenadas del punto I.
        xi=xp-h*math.cos(phi+(3*math.pi/2))
        yi=yp-h*math.sin(phi+(3*math.pi/2))
            
        # Primera cadena vectorial 1.
        L_GA=math.sqrt((xg-xa)**2+(yg-ya)**2) #REVISADO
        gamma1=math.acos((L_GA**2+L_DA**2-L_GD**2)/(2*L_GA*L_DA))#REVISADO
        fi1=math.atan2((yg-ya),(xg-xa))# Revisado
        tetha1=fi1+gamma1# revisado
            
        # Segunda cadena vectorial 2.
        L_HB=math.sqrt((xh-xb)**2+(yh-yb)**2)# Revisado
        gamma2=math.acos((L_HB**2+L_EB**2-L_HE**2)/(2*L_EB*L_HB))# Revisado 
        fi2=math.atan2((yh-yb),(xh-xb))#revisado
        tetha2=fi2+gamma2 #revisado
            
        # Tercera cadena vetorial 3.
        L_IC=math.sqrt((xi-xc)**2+(yi-yc)**2) #Revisado
        gamma3=math.acos((L_IC**2+L_FC**2-L_IF**2)/(2*L_FC*L_IC)) #Revisado
        fi3=math.atan2((yi-yc),(xi-xc)) #revisado
        tetha3=fi3+gamma3 #revisado
            
        tetha11=tetha1*180/math.pi
            
        tetha22=tetha2*180/math.pi
            
        tetha33=tetha3*180/math.pi
            
        # Luego se inicia el factor de conversion (Grados-->Bytes) para que los
        # servor puedan iniciar la lectura y establecer inicio de movimiento.
            
        # Para dar inicio al movimiento se establece la pose inicial de la
        # plataforma movil.
            
        # cadena 1
        if tetha11<=0:
            tetha11=tetha11-30
        elif tetha11>=0:
            tetha11=tetha11-30
            
    #     cadena 2
        if tetha22<=0:
            tetha22=tetha22+45       
        elif tetha22>=0:
            tetha22=tetha22+45
        
    #     cadena 3
        if tetha33<=0:
            tetha33=tetha33+65
        elif tetha33>=0:
            tetha33=tetha33+65
        
    #    Ajuste por minimos cuadrados (Error)
        
        m=0.99182076813655761024
        b=-10.9331436699857752489
        tetha11=m*tetha11+b
        tetha11=tetha11+19
    
    #     para el servo 2
    
        m2=0.98968705547652916074
        b2=-8.4679943100995732575
        tetha22=m2*tetha22+b2
        tetha22=tetha22+17
    
    #     para el servo 3
        m3=0.99497392128971076339
        b3=-3.7439544807965860597
        tetha33=m3*tetha33+b3
        tetha33=tetha33+6.5
        
      
    #     Inicio de conversion - Esta parte envia el dato en bytes al driver.  
        #Se convierten a decimal para poder hacer luego la conversión a bytes en el serial
        Btheta11 = round((tetha11*decimal)/grados)
        Btheta22 = round((tetha22*decimal)/grados)
        Btheta33 = round((tetha33*decimal)/grados)
        
    
        return Btheta11, Btheta22, Btheta33
    
    #---------------------------------------------------------------------------------------------------
    
    def abrir_puerto_serial(self, puerto='COM3', tasa_bs= 1000000, paridad=1, rtscts=1, timeout=0):
        """Este metodo abre el puerto serial especificado en el argumento con las carc¿acteristicas de comunicacion de los motores"""
        
        ser = serial.Serial(
        port=puerto,
        baudrate=tasa_bs,
        parity=serial.PARITY_NONE,
        rtscts=rtscts,
        timeout=timeout
#    stopbits=1,
#    bytesize=8
        )
        return ser
    
#------------------------------------------------------------------------------------------------
    
    def cerrar_puerto_serial(self, ser):
        """Este metodo cierra el puerto serial"""
        
        ser.close()
        return None
    
    
#------------------------------------------------------------------------------------------------
    
    def enviar_tethas_servomotores(self, ser, b1, b2, b3):
        """Este metodo envía los angulos betha a los servomotores correspondientes"""
        
        betha = ([b1, b2, b3])
        bin_betha1 = bin(betha[0])
        bin_betha2 = bin(betha[1])
        bin_betha3 = bin(betha[2])
        
        bin_betha1 = "0000000000" + bin_betha1[2:(len(bin_betha1))]
        bin_betha2 = "0000000000" + bin_betha2[2:(len(bin_betha2))]
        bin_betha3 = "0000000000" + bin_betha3[2:(len(bin_betha3))]
        
        nl_bin_betha1 = bin_betha1[(len(bin_betha1)-8):(len(bin_betha1))]
        nh_bin_betha1 = bin_betha1[(len(bin_betha1)-12):(len(bin_betha1)-8)]
        
        nl_bin_betha2 = bin_betha2[(len(bin_betha2)-8):(len(bin_betha2))]
        nh_bin_betha2 = bin_betha2[(len(bin_betha2)-12):(len(bin_betha2)-8)]
        
        nl_bin_betha3 = bin_betha3[(len(bin_betha3)-8):(len(bin_betha3))]
        nh_bin_betha3 = bin_betha3[(len(bin_betha3)-12):(len(bin_betha3)-8)]
        
        
        nl_dec_betha1 = int(nl_bin_betha1,2)
        nh_dec_betha1 = int(nh_bin_betha1,2)
        
        nl_dec_betha2 = int(nl_bin_betha2,2)
        nh_dec_betha2 = int(nh_bin_betha2,2)
        
        nl_dec_betha3 = int(nl_bin_betha3,2)
        nh_dec_betha3 = int(nh_bin_betha3,2)
        
        #print(nl_dec_betha2)
        #print(nh_dec_betha2)
        
        r1 = 1+5+3+30+nl_dec_betha1+nh_dec_betha1
        r2 = 2+5+3+30+nl_dec_betha2+nh_dec_betha2
        r3 = 3+5+3+30+nl_dec_betha3+nh_dec_betha3
        
        z1 = bin(r1)
        z1 = "0000000000" + z1[2:(len(z1))]
        z2 = bin(r2)
        z2 = "0000000000" + z2[2:(len(z2))]
        z3 = bin(r3)
        z3 = "0000000000" + z3[2:(len(z3))]
        
        z1 = z1[(len(z1)-12):(len(z1))]
        z2 = z2[(len(z2)-12):(len(z2))]
        z3 = z3[(len(z3)-12):(len(z3))]
        #print(z1)
        
        
        not_z1 = bin(int(z1,2)^4095)
        not_z2 = bin(int(z2,2)^4095)
        not_z3 = bin(int(z3,2)^4095)
#        print(not_z1)
        
        n11 = not_z1[(len(not_z1)-8):(len(not_z1))]
        n22 = not_z2[(len(not_z2)-8):(len(not_z2))]
        n33 = not_z3[(len(not_z3)-8):(len(not_z3))]
#        print(n11)
        
        p11 = int(n11,2) #este seria el checksum
        p22 = int(n22,2)
        p33 = int(n33,2)
        
        #Parametro para enviarposicion a losservos
        parametro = 30
        
#        vector= [255,255,1,5,3,parametro,nl_dec_betha1,nh_dec_betha1,p11,255,255,2,5,3,parametro,nl_dec_betha2,nh_dec_betha2,p22,255,255,3,5,3,parametro,nl_dec_betha3,nh_dec_betha3,p33]
        vector_1= [255,255,1,5,3,parametro,nl_dec_betha1,nh_dec_betha1,p11]
        vector_2= [255,255,2,5,3,parametro,nl_dec_betha2,nh_dec_betha2,p22]
        vector_3= [255,255,3,5,3,parametro,nl_dec_betha3,nh_dec_betha3,p33]
        
        ser.write(vector_1)
        ser.write(vector_2)
        ser.write(vector_3)
        
        mess_status = ser.read(1000)

        return mess_status
    
    #------------------------------------------------------------------------------------------------
    
    def inicializar_servos(self,ser):
        """Este metodo envía los angulos betha a los servomotores correspondientes"""
        #Parametro para enviarposicion a losservos
        parametro = 32
        betha = ([9, 9, 9])
        bin_betha1 = bin(betha[0])
        bin_betha2 = bin(betha[1])
        bin_betha3 = bin(betha[2])
        
        bin_betha1 = "0000000000" + bin_betha1[2:(len(bin_betha1))]
        bin_betha2 = "0000000000" + bin_betha2[2:(len(bin_betha2))]
        bin_betha3 = "0000000000" + bin_betha3[2:(len(bin_betha3))]
        
        nl_bin_betha1 = bin_betha1[(len(bin_betha1)-8):(len(bin_betha1))]
        nh_bin_betha1 = bin_betha1[(len(bin_betha1)-12):(len(bin_betha1)-8)]
        
        nl_bin_betha2 = bin_betha2[(len(bin_betha2)-8):(len(bin_betha2))]
        nh_bin_betha2 = bin_betha2[(len(bin_betha2)-12):(len(bin_betha2)-8)]
        
        nl_bin_betha3 = bin_betha3[(len(bin_betha3)-8):(len(bin_betha3))]
        nh_bin_betha3 = bin_betha3[(len(bin_betha3)-12):(len(bin_betha3)-8)]
        
        
        nl_dec_betha1 = int(nl_bin_betha1,2)
        nh_dec_betha1 = int(nh_bin_betha1,2)
        
        nl_dec_betha2 = int(nl_bin_betha2,2)
        nh_dec_betha2 = int(nh_bin_betha2,2)
        
        nl_dec_betha3 = int(nl_bin_betha3,2)
        nh_dec_betha3 = int(nh_bin_betha3,2)
        
        #print(nl_dec_betha2)
        #print(nh_dec_betha2)
        
        r1 = 1+5+3+parametro+nl_dec_betha1+nh_dec_betha1
        r2 = 2+5+3+parametro+nl_dec_betha2+nh_dec_betha2
        r3 = 3+5+3+parametro+nl_dec_betha3+nh_dec_betha3
        
        z1 = bin(r1)
        z1 = "0000000000" + z1[2:(len(z1))]
        z2 = bin(r2)
        z2 = "0000000000" + z2[2:(len(z2))]
        z3 = bin(r3)
        z3 = "0000000000" + z3[2:(len(z3))]
        
        z1 = z1[(len(z1)-12):(len(z1))]
        z2 = z2[(len(z2)-12):(len(z2))]
        z3 = z3[(len(z3)-12):(len(z3))]
        #print(z1)
        
        
        not_z1 = bin(int(z1,2)^4095)
        not_z2 = bin(int(z2,2)^4095)
        not_z3 = bin(int(z3,2)^4095)
#        print(not_z1)
        
        n11 = not_z1[(len(not_z1)-8):(len(not_z1))]
        n22 = not_z2[(len(not_z2)-8):(len(not_z2))]
        n33 = not_z3[(len(not_z3)-8):(len(not_z3))]
#        print(n11)
        
        p11 = int(n11,2) #este seria el checksum
        p22 = int(n22,2)
        p33 = int(n33,2)
        
        
        
#        vector= [255,255,1,5,3,parametro,nl_dec_betha1,nh_dec_betha1,p11,255,255,2,5,3,parametro,nl_dec_betha2,nh_dec_betha2,p22,255,255,3,5,3,parametro,nl_dec_betha3,nh_dec_betha3,p33]
        vector_1= [255,255,1,5,3,parametro,nl_dec_betha1,nh_dec_betha1,p11]
        vector_2= [255,255,2,5,3,parametro,nl_dec_betha2,nh_dec_betha2,p22]
        vector_3= [255,255,3,5,3,parametro,nl_dec_betha3,nh_dec_betha3,p33]
        
        ser.write(vector_1)
        ser.write(vector_2)
        ser.write(vector_3)
        
        mess_status = ser.read(1000)
        
        
        return mess_status
    
    
    #------------------------------------------------------------------------------------------------
    
    
    
#===========================================================================        
def main():
    """Funcion principal
       Solo se debe ejecutar si se ejecuta el programa de forma individual
       Pero no se debe ejecutar si se carga como modulo dentro de otro programa
    """

    a=80
    b=-100
    #Coordenadas de busqueda las marcas
    marcas_cad_mm = ([[150,-50],[310,-50],[130,0],[330,0],[170,50],[290,50],[100,100],[360,100],[250,145],[435,145],[80,220],[380,220],[170,250],[290,250],[80,280],[380,280],[130,330],[300,330],[230,318.4]])
    marcas_cad_mm_1 = ([[150+a,650+b],[130+a,600+b],[330+a,600+b],[170+a,550+b],[290+a,550+b],[100+a,500+b],[360+a,500+b],[80+a,380+b],[380+a,380+b],[170+a,350+b],[80+a,320+b],[380+a,320+b],[130+a,270+b],[230+a,281.6+b]])
    marcas_cad_mm_neg = ([[0,0],[150,50],[310,50],[130,0],[330,0],[170,-50],[290,-50],[100,-100],[360,-100],[25,-145],[435,-145],[80,-220],[380,-220],[170,-250],[290,-250],[80,-280],[380,-280],[130,-330],[300,-330],[230,-318.4]])
    marcas_click = ([[248,428],[359,431],[238,389],[378,392],[276,354],[340,355],[223,322],[396,323],[176,288],[426,291],[200,234],[412,246],[282,219],[342,224],[203,189],[417,198],[244,180],[381,184],[311,173]])
    centroides_marcas = ([[247,426],[237,388],[376,390],[276,353],[339,354],[222,320.5],[395,322],[199,233],[411,245],[280.5,218],[202,188],[416,197],[242,179],[310,172]])
    
    referencia_marca_17_medido=[201,195]
    referencia_marca_17_cad=[130,330]
    nuevo_cero = [71,525]
    
    #Se utiliza esta linea si se desea probar e manipulador con un video precargado
#    cap = cv2.VideoCapture('videoFinalM.wmv')
    
    #Se utiliza esta linea si se desea probar directamente con la camara. Hay que especificar el numero de la camara en el sistema
    cap = cv2.VideoCapture(0)
    
    #Se captura el valor de la tasa de adquisicion del video para alimentar Kalman. Solo para video
#    fps = cap.get(cv2.CAP_PROP_FPS)
    
    #Se especifica la tasa de captura para Kalman. ¿Como se hace?
#    fps = 30
    
    # si no se conce la ubicacion de las marcas, se pueden indicar con la siguiente linea
    #marcas_click = prm.inicializar_marcas(img_base)
    
    #Se crea el objeto e la clase proyectividad
    estabilizador = ProyectividadOpenCV()
    
    #Se carga una imagen base para hallar la homografia de esta contra el CAD y asi utilizarla como base
    img_for_mm = cv2.imread("img_base.png")
    
    #Este metodo halla la homografia contra el cad desde una serie de puntos correspondientes a los centros estimados. El se encarga de buscar los centroides
    #img_base = estabilizador.estabilizar_desde_marcas(img_for_mm,marcas_click,marcas_cad_mm_1)
      
    #Este metodo halla la homografia contra el cad desde una serie de puntos correspondientes a los centroides de las marcas
    img_base = estabilizador.estabilizar_desde_centroides_marcas(img_for_mm,centroides_marcas,marcas_cad_mm_1)
    
    #Se crea una variable delta de t para kalman
    delta_t = 0.1
    
    #Se mide el tiempo que pasa entre la captura de un frame y otro
    tiempo_inicial = time()
    
    #Se inicializa Kalman
    cx, cy = 200, 200    
    kalman = cv2.KalmanFilter(4,2)    
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)    
    kalman.transitionMatrix = np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]],np.float32)    
    kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.01    
    kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.0001
    tp = [23,13]
    
    #Se abre el puerto serial y se deja abierto para una comunicacion constante
    ser = estabilizador.abrir_puerto_serial(puerto='COM6')#Hay que verificar el puerto
    if ser.isOpen():
        estabilizador.inicializar_servos(ser)
    
#    ser = serial.Serial(
#    port='COM5',
#    baudrate=921600,
#    parity=serial.PARITY_EVEN,
#    rtscts=1,
#    timeout=0
##    stopbits=1,
##    bytesize=8
#    )
    tiempo_inicial = 0
    tiempo_final = 0.1
    while(True):
        tiempo_final = time()
        delta_t = tiempo_final-tiempo_inicial
        tiempo_inicial = tiempo_final
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        print(delta_t)
        
        #Esta clase estabiliza automaticamente la imagen con base una imagen inicial
        estabilizada = estabilizador.estabilizador_imagen(frame,img_base)
        
        #Se aplica un ruido gausiano para suavizar bordes
        blur = cv2.blur(estabilizada, (3,3))
        #Se hace la transformacion a HSV
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        #Se aplica la mascara de color que solo deja pasar rojos. Video
#        thresh_objeto = cv2.inRange(hsv,np.array((0,50,50)), np.array((10,255,255)))
        
#        Se aplica la mascara de color que solo deja pasar rojos. Camara
        thresh_objeto = cv2.inRange(hsv,np.array((160,100,100)), np.array((179,255,255)))
        
        # se buscan los contornos en la imagen filtada para rojos
        image, contours,hierarchy = cv2.findContours(thresh_objeto,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        #Se inicializa la variable para contar los frames en los que se pierde el objeto para alimentar Kalman
        cont_frame = 1
    
        # Se hace la busqueda del objeto y se filtra por area para determinar que sea el adecuado
        max_area = 550
        best_cnt = 1
        for cnt in contours:
            area = 1
            area = cv2.contourArea(cnt)
    #        print(area)
            if area > max_area and area < 750:
                max_area = area
                best_cnt = cnt
                # Si seencuentra un area que cumpla, entonces, se halla el centroide de esta
                M = cv2.moments(best_cnt)
                cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                
                #Se dibuja una cruz verde sobre el objeto encontrado
                cv2.line(blur,(cx-10,cy),(cx+10,cy),(0,255,0),1)
                cv2.line(blur,(cx,cy-10),(cx,cy+10),(0,255,0),1)
                
                # Se hace la conversion del formato para alimentar Kalman con el centroide
                mp = np.array([[np.float32(cx)],[np.float32(cy)]])
                tp = kalman.predict()
                kalman.correct(mp)
                
                #Como el objeto fue encontrado, entonces, se deja en 1 el conteo de frames
                cont_frame = 1
            
            # Si no seencuentra el objeto, se hace la estimacion de Kalman y no se actualiza la medida
            if area <= max_area or area >= 750: 
                kalman.transitionMatrix = np.array([[1,0,cont_frame*delta_t,0],[0,1,0,cont_frame*delta_t],[0,0,1,0],[0,0,0,1]],np.float32)
                
                #Se hace el conteo de los frames en los que no se encuentra la marca para alimentar Kalman
                cont_frame = cont_frame + 1
                
                #Sedibuja una cruz azul en la posicion estimada del objeto
                tp = kalman.predict()
                cv2.line(blur,(tp[0]-10,tp[1]),(tp[0]+10,tp[1]),(255,0,0),1)
                cv2.line(blur,(tp[0],tp[1]-10),(tp[0],tp[1]+10),(255,0,0),1)
                
     
        # Se halla la varianza de la estimacion para reducir la zona de busqueda en el siguiente frame
        varianza_x = kalman.errorCovPost[0,0]
        varianza_y = kalman.errorCovPost[1,1]
        devstd_x = varianza_x**0.5
        devstd_y = varianza_y**0.5
      
        #para 6 desviaciones estandar
        marco_x=devstd_x*6
        marco_y=devstd_y*6
        
        # Se dibuja un circulo blanco que crece con la varianza a razon de 6 desviaciones estandar
        cv2.circle(blur, (int(tp[0][0]),int(tp[1][0])), int(marco_x), (255,255,255),1, cv2.LINE_AA)
#        print(tp)
        
        #Se obtienen las coordenadas en centimetros para la cinematicainversa
        cor_x=10+(tp[0][0]-nuevo_cero[0])/10
        cor_y=(nuevo_cero[1]-tp[1][0])/10
#        print(cor_x,cor_y)
        if cor_x <= 14 or cor_x >= 34:
            cor_x = 23
        if cor_y <= 4 or cor_y >= 26:
            cor_y = 13.28
            
        
#        print(cor_x,cor_y)
        #Se utiliza la cinematica inversa para obtener el valr de los angulos en decimal de 0 a 4096
        b1, b2, b3 = estabilizador.cinematica_inversa(cor_x,cor_y)
##        print(tp[0]/10,tp[1]/10)
#        print(angulos_decimales)
        
        if ser.isOpen():
            mensaje_status = estabilizador.enviar_tethas_servomotores(ser,b1, b2, b3)
        
#        print(mensaje_status)
        #Visualizacion de imagenes
        cv2.imshow('Umbral',blur)
        cv2.imshow('Mask',thresh_objeto)
    #    cv2.imshow('Marcas',thresh_marcas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    ser.close()
    cap.release()
    cv2.destroyAllWindows()
    

#===========================================================================
if __name__ == '__main__':
    main()
        
