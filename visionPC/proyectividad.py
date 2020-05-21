#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insert module description here

Created on Wed Sep 13 18:07:14 2017
@author: jolumartinez
"""

# Imports

import cv2
import numpy as np
import os.path
# import serial
import math
from scipy.interpolate import interp1d
from time import time
import numpy as np
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
import open3d as o3d

puntos_click = list()


# ================================================================
def click_and_count(event, x, y, flags, param):
    """Definicion del callback que atiende el mouse"""

    global puntos_click

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        puntos_click.append((x, y))


# +=================================================================================================
class ProyectividadOpenCV():
    """
    Esta es una clase para solucionar problemas con homografias
    """
    # Atributos de la clase
    error_reproyeccion = 4

    # --------------------------------------------------------------------------
    def __init__(self):
        """Inicializador del objeto miembro de la clase"""

    # --------------------------------------------------------------------------
    def coser_imagenes(self, ruta_img_base, ruta_img_adicional, radio=0.75, error_reproyeccion=4.0,
                       coincidencias=False):
        """Metodo que carga una imagen desde una ruta en disco duro"""

        imagen_adicional = ruta_img_adicional
        imagen_base = ruta_img_base
        # Se obtienen los puntos deinterés

        (kpsBase, featuresBase) = self.obtener_puntos_interes(imagen_base)
        (kpsAdicional, featuresAdicional) = self.obtener_puntos_interes(imagen_adicional)
        # Se buscan las coincidencias

        M = self.encontrar_coincidencias(imagen_base, imagen_adicional, kpsBase, kpsAdicional, featuresBase,
                                         featuresAdicional, radio)

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

    # --------------------------------------------------------------------------
    def estabilizador_imagen(self, imagen_base, imagen_a_estabilizar, radio=0.75, error_reproyeccion=4.0,
                             coincidencias=False):
        """Esta clase devuelve una secuencia de imágenes tomadas de la cámara estabilizada con respecto a la primera imagen"""

        # Se obtienen los puntos deinterés

        (kpsBase, featuresBase) = self.obtener_puntos_interes(imagen_base)
        (kpsAdicional, featuresAdicional) = self.obtener_puntos_interes(imagen_a_estabilizar)
        # Se buscan las coincidencias        

        M = self.encontrar_coincidencias(imagen_base, imagen_a_estabilizar, kpsBase, kpsAdicional, featuresBase,
                                         featuresAdicional, radio)

        if M is None:
            print("pocas coincidencias")
            return None

        if len(M) > 4:
            # construct the two sets of points

            #            M2 = cv2.getPerspectiveTransform(ptsA,ptsB)
            (H, status) = self.encontrar_H_RANSAC_Estable(M, kpsBase, kpsAdicional, error_reproyeccion)
            estabilizada = cv2.warpPerspective(imagen_base, H, (imagen_base.shape[1], imagen_base.shape[0]))
            return estabilizada
        print("sin coincidencias")
        return None

    def img_alignment_sequoia(self, img_RGB, img_GRE, img_base_NIR, img_RED, img_REG, width, height):
        """This class takes the five images given by Sequoia Camera and makes a photogrammetric
        alignment. Returns four images (GRE, NIR, RED, REG) aligned with the RGB image"""

        # Se valida que si estén todas las variables en el argumento

        # width, height = img_SIZE

        # Se redimencionan todas las imagenes al mismo tamaño especificado en image_SIZE

        b_RGB = cv2.resize(img_RGB, (width, height), interpolation=cv2.INTER_LINEAR)
        b_GRE = cv2.resize(img_GRE, (width, height), interpolation=cv2.INTER_LINEAR)
        base_NIR = cv2.resize(img_base_NIR, (width, height), interpolation=cv2.INTER_LINEAR)
        b_RED = cv2.resize(img_RED, (width, height), interpolation=cv2.INTER_LINEAR)
        b_REG = cv2.resize(img_REG, (width, height), interpolation=cv2.INTER_LINEAR)

        # Se estabilizan todas las imágenes con respecto a la imagen base

        stb_GRE = self.estabilizador_imagen(b_GRE, base_NIR)
        stb_RGB = self.estabilizador_imagen(b_RGB, base_NIR)
        stb_RED = self.estabilizador_imagen(b_RED, base_NIR)
        stb_REG = self.estabilizador_imagen(b_REG, base_NIR)

        return stb_RGB, stb_GRE, base_NIR, stb_RED, stb_REG

    # --------------------------------------------------------------------------
    def obtener_puntos_interes(self, imagen):
        """Se obtienen los puntos de interes cn SIFT"""

        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(imagen, None)

        return kps, features

    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    def encontrar_H_marcas(self, las_xprima, las_x):
        """Metodo para estimar la homografia"""
        # Se utiliza 0 y no RANSAC porque deseo que utilice todos los puntos que se tienen
        H, estado = cv2.findHomography(las_x, las_xprima, 0, 0.1)
        return H, estado

    # --------------------------------------------------------------------------
    def estabilizar_desde_marcas(self, imagen, marcas_click, marcas_cad_mm):
        """Esta clase retorna una imagen estabilizada con base en una imagen abstraida delas marcas del cad dadas en mm"""

        # Lo primero es tratar la imagen entrante
        blur = cv2.blur(imagen, (3, 3))
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Se aplica un filtro de color verde para separar las marcas del fondo
        thresh_marcas = cv2.inRange(hsv, np.array((49, 50, 50)), np.array((107, 255, 255)))
        marcas_H = list()
        cad_H = list()
        # Se hace una busqueda de las marcas visibles en un radio de 30 pixelesy se filtranpor area para sacar los pares que permitiran hallar la homografia
        for i in range(0, len(marcas_click)):
            #            print(i)
            x_men = marcas_click[i][0] - 10
            x_may = marcas_click[i][0] + 10
            y_men = marcas_click[i][1] - 10
            y_may = marcas_click[i][1] + 10
            area_marca = thresh_marcas[y_men:y_may, x_men:x_may]
            image_marcas, contours_marcas, hierarchy_marcas = cv2.findContours(area_marca, cv2.RETR_LIST,
                                                                               cv2.CHAIN_APPROX_SIMPLE)

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
                    cx, cy = int(cM['m10'] / cM['m00']), int(cM['m01'] / cM['m00'])
                    cx = x_men + cx
                    cy = y_men + cy
                    marcas_H.append((cx, cy))
                    cad_H.append((marcas_cad_mm[i][0] + 100, marcas_cad_mm[i][1]))
        #                    print(marcas_H,cad_H)

        las_x = np.array(cad_H)
        las_xprima = np.array(marcas_H)
        print(las_x, las_xprima)
        if len(las_xprima) > 4:
            H, estado = self.encontrar_H_marcas(las_x, las_xprima)

            estabilizada = cv2.warpPerspective(imagen, H, (1200, 1200))

            return estabilizada

        return None

    # --------------------------------------------------------------------------
    def estabilizar_desde_centroides_marcas(self, imagen, marcas_click, marcas_cad_mm):
        """Esta clase retorna una imagen estabilizada con base en una imagen abstraida delas marcas del cad dadas en mm"""

        las_x = np.array(marcas_cad_mm)
        las_xprima = np.array(marcas_click)
        print(las_x, las_xprima)
        if len(las_xprima) > 4:
            H, estado = self.encontrar_H_marcas(las_x, las_xprima)

            estabilizada = cv2.warpPerspective(imagen, H, (650, 650))

            return estabilizada

        return None

    # --------------------------------------------------------------------------
    def inicializar_marcas(self, img_base):
        """Permite al usuario hacer click en el centro apriximado de cada marca y las guarda en orden"""

        global puntos_click

        # Copiar la imagen original para poder escribir sobre ella
        # Sin modificarla
        imagen_conmarcas = self.img_base.copy()

        # Mostrar la imagen
        cv2.namedWindow("Imagen_base")
        cv2.setMouseCallback("Imagen_base", click_and_count)

        while True:
            # Mostrar a imagen
            cv2.imshow("Imagen_base", imagen_conmarcas)
            key = cv2.waitKey(1) & 0xFF

            # Menu principal
            # Si se presiona r resetee la imagen
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
                    # Coordenadas
                    x, y = coords[0], coords[1]
                    # Dibujar un circulo
                    cv2.circle(imagen_conmarcas, (x, y), 5, (0, 0, 255), 5, 2)
                    # Seleccionar una fuente
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(imagen_conmarcas, str(pts_id + 1), (x, y), font, 10, (0, 0, 255), 5)

    # ------------------------------------------------------------------------------------------------

    def cinematica_inversa(self, a, b):
        """Recibe las coordenadas en centimetros"""
        # Se valida que la coordenada se encuentre dentro del area de trabajo. Si no esta se ubica en el centro el manipulador
        if a <= 14 or a >= 34:
            a = 15
        if b <= 4 or b >= 26:
            b = 16

        xp = a
        yp = b

        xa = 0
        ya = 0
        xb = 46
        yb = 0
        xc = 46 / 2
        yc = 46 * math.sin(math.pi / 3)

        # Dimenciones de los eslabones y acopladores del manipulador (Unidades en cm.) 
        manivela = 19

        # Eslabones (Cadenas Cinematica 1)
        l1 = manivela
        L_DA = l1
        l4 = manivela
        L_GD = l4

        # Eslabones (Cadena Cinematica 2)
        l2 = manivela
        L_EB = l2
        l5 = manivela
        L_HE = l5

        # Eslabones (Cadena Cinematica 3)
        l3 = manivela
        L_FC = l3
        l6 = manivela
        L_IF = l6

        # plataforma movil
        # El parametro (h) representa el baricentro de la plataforma.
        h = 7.5056
        # phi=0; % Plataforma no rota.
        phi = 0 * math.pi / 180  # Plataforma rota.

        # Datos de conversión (Grados --> Bits)

        grados = 360
        decimal = 4096
        #    Resolucion del servomotor.

        #        xp=23#               23.5
        #        yp=13.279056191#       13.279056191

        # Coordenadas de la plataforma movil 
        # Coordenadas del punto G.
        xg = xp - h * math.cos(phi + (math.pi / 6))
        yg = yp - h * math.sin(phi + (math.pi / 6))
        # Coordenadas del punto H.
        xh = xp - h * math.cos(phi + (math.pi - (math.pi / 6)))
        yh = yp - h * math.sin(phi + (math.pi - (math.pi / 6)))

        # Coodenadas del punto I.
        xi = xp - h * math.cos(phi + (3 * math.pi / 2))
        yi = yp - h * math.sin(phi + (3 * math.pi / 2))

        # Primera cadena vectorial 1.
        L_GA = math.sqrt((xg - xa) ** 2 + (yg - ya) ** 2)  # REVISADO
        gamma1 = math.acos((L_GA ** 2 + L_DA ** 2 - L_GD ** 2) / (2 * L_GA * L_DA))  # REVISADO
        fi1 = math.atan2((yg - ya), (xg - xa))  # Revisado
        tetha1 = fi1 + gamma1  # revisado

        # Segunda cadena vectorial 2.
        L_HB = math.sqrt((xh - xb) ** 2 + (yh - yb) ** 2)  # Revisado
        gamma2 = math.acos((L_HB ** 2 + L_EB ** 2 - L_HE ** 2) / (2 * L_EB * L_HB))  # Revisado
        fi2 = math.atan2((yh - yb), (xh - xb))  # revisado
        tetha2 = fi2 + gamma2  # revisado

        # Tercera cadena vetorial 3.
        L_IC = math.sqrt((xi - xc) ** 2 + (yi - yc) ** 2)  # Revisado
        gamma3 = math.acos((L_IC ** 2 + L_FC ** 2 - L_IF ** 2) / (2 * L_FC * L_IC))  # Revisado
        fi3 = math.atan2((yi - yc), (xi - xc))  # revisado
        tetha3 = fi3 + gamma3  # revisado

        tetha11 = tetha1 * 180 / math.pi

        tetha22 = tetha2 * 180 / math.pi

        tetha33 = tetha3 * 180 / math.pi

        # Luego se inicia el factor de conversion (Grados-->Bytes) para que los
        # servor puedan iniciar la lectura y establecer inicio de movimiento.

        # Para dar inicio al movimiento se establece la pose inicial de la
        # plataforma movil.

        # cadena 1
        if tetha11 <= 0:
            tetha11 = tetha11 - 30
        elif tetha11 >= 0:
            tetha11 = tetha11 - 30

        #     cadena 2
        if tetha22 <= 0:
            tetha22 = tetha22 + 45
        elif tetha22 >= 0:
            tetha22 = tetha22 + 45

        #     cadena 3
        if tetha33 <= 0:
            tetha33 = tetha33 + 65
        elif tetha33 >= 0:
            tetha33 = tetha33 + 65

        #    Ajuste por minimos cuadrados (Error)

        m = 0.99182076813655761024
        b = -10.9331436699857752489
        tetha11 = m * tetha11 + b
        tetha11 = tetha11 + 19

        #     para el servo 2

        m2 = 0.98968705547652916074
        b2 = -8.4679943100995732575
        tetha22 = m2 * tetha22 + b2
        tetha22 = tetha22 + 17

        #     para el servo 3
        m3 = 0.99497392128971076339
        b3 = -3.7439544807965860597
        tetha33 = m3 * tetha33 + b3
        tetha33 = tetha33 + 6.5

        #     Inicio de conversion - Esta parte envia el dato en bytes al driver.
        # Se convierten a decimal para poder hacer luego la conversión a bytes en el serial
        Btheta11 = round((tetha11 * decimal) / grados)
        Btheta22 = round((tetha22 * decimal) / grados)
        Btheta33 = round((tetha33 * decimal) / grados)

        return Btheta11, Btheta22, Btheta33

    # ---------------------------------------------------------------------------------------------------

    def abrir_puerto_serial(self, puerto='COM3', tasa_bs=1000000, paridad=1, rtscts=1, timeout=0):
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

    # ------------------------------------------------------------------------------------------------

    def cerrar_puerto_serial(self, ser):
        """Este metodo cierra el puerto serial"""

        ser.close()
        return None

    # ------------------------------------------------------------------------------------------------

    def enviar_tethas_servomotores(self, ser, b1, b2, b3):
        """Este metodo envía los angulos betha a los servomotores correspondientes"""

        betha = ([b1, b2, b3])
        bin_betha1 = bin(betha[0])
        bin_betha2 = bin(betha[1])
        bin_betha3 = bin(betha[2])

        bin_betha1 = "0000000000" + bin_betha1[2:(len(bin_betha1))]
        bin_betha2 = "0000000000" + bin_betha2[2:(len(bin_betha2))]
        bin_betha3 = "0000000000" + bin_betha3[2:(len(bin_betha3))]

        nl_bin_betha1 = bin_betha1[(len(bin_betha1) - 8):(len(bin_betha1))]
        nh_bin_betha1 = bin_betha1[(len(bin_betha1) - 12):(len(bin_betha1) - 8)]

        nl_bin_betha2 = bin_betha2[(len(bin_betha2) - 8):(len(bin_betha2))]
        nh_bin_betha2 = bin_betha2[(len(bin_betha2) - 12):(len(bin_betha2) - 8)]

        nl_bin_betha3 = bin_betha3[(len(bin_betha3) - 8):(len(bin_betha3))]
        nh_bin_betha3 = bin_betha3[(len(bin_betha3) - 12):(len(bin_betha3) - 8)]

        nl_dec_betha1 = int(nl_bin_betha1, 2)
        nh_dec_betha1 = int(nh_bin_betha1, 2)

        nl_dec_betha2 = int(nl_bin_betha2, 2)
        nh_dec_betha2 = int(nh_bin_betha2, 2)

        nl_dec_betha3 = int(nl_bin_betha3, 2)
        nh_dec_betha3 = int(nh_bin_betha3, 2)

        # print(nl_dec_betha2)
        # print(nh_dec_betha2)

        r1 = 1 + 5 + 3 + 30 + nl_dec_betha1 + nh_dec_betha1
        r2 = 2 + 5 + 3 + 30 + nl_dec_betha2 + nh_dec_betha2
        r3 = 3 + 5 + 3 + 30 + nl_dec_betha3 + nh_dec_betha3

        z1 = bin(r1)
        z1 = "0000000000" + z1[2:(len(z1))]
        z2 = bin(r2)
        z2 = "0000000000" + z2[2:(len(z2))]
        z3 = bin(r3)
        z3 = "0000000000" + z3[2:(len(z3))]

        z1 = z1[(len(z1) - 12):(len(z1))]
        z2 = z2[(len(z2) - 12):(len(z2))]
        z3 = z3[(len(z3) - 12):(len(z3))]
        # print(z1)

        not_z1 = bin(int(z1, 2) ^ 4095)
        not_z2 = bin(int(z2, 2) ^ 4095)
        not_z3 = bin(int(z3, 2) ^ 4095)
        #        print(not_z1)

        n11 = not_z1[(len(not_z1) - 8):(len(not_z1))]
        n22 = not_z2[(len(not_z2) - 8):(len(not_z2))]
        n33 = not_z3[(len(not_z3) - 8):(len(not_z3))]
        #        print(n11)

        p11 = int(n11, 2)  # este seria el checksum
        p22 = int(n22, 2)
        p33 = int(n33, 2)

        # Parametro para enviarposicion a losservos
        parametro = 30

        #        vector= [255,255,1,5,3,parametro,nl_dec_betha1,nh_dec_betha1,p11,255,255,2,5,3,parametro,nl_dec_betha2,nh_dec_betha2,p22,255,255,3,5,3,parametro,nl_dec_betha3,nh_dec_betha3,p33]
        vector_1 = [255, 255, 1, 5, 3, parametro, nl_dec_betha1, nh_dec_betha1, p11]
        vector_2 = [255, 255, 2, 5, 3, parametro, nl_dec_betha2, nh_dec_betha2, p22]
        vector_3 = [255, 255, 3, 5, 3, parametro, nl_dec_betha3, nh_dec_betha3, p33]

        ser.write(vector_1)
        ser.write(vector_2)
        ser.write(vector_3)

        mess_status = ser.read(1000)

        return mess_status

    # ------------------------------------------------------------------------------------------------

    def inicializar_servos(self, ser):
        """Este metodo envía los angulos betha a los servomotores correspondientes"""
        # Parametro para enviarposicion a losservos
        parametro = 32
        betha = ([9, 9, 9])
        bin_betha1 = bin(betha[0])
        bin_betha2 = bin(betha[1])
        bin_betha3 = bin(betha[2])

        bin_betha1 = "0000000000" + bin_betha1[2:(len(bin_betha1))]
        bin_betha2 = "0000000000" + bin_betha2[2:(len(bin_betha2))]
        bin_betha3 = "0000000000" + bin_betha3[2:(len(bin_betha3))]

        nl_bin_betha1 = bin_betha1[(len(bin_betha1) - 8):(len(bin_betha1))]
        nh_bin_betha1 = bin_betha1[(len(bin_betha1) - 12):(len(bin_betha1) - 8)]

        nl_bin_betha2 = bin_betha2[(len(bin_betha2) - 8):(len(bin_betha2))]
        nh_bin_betha2 = bin_betha2[(len(bin_betha2) - 12):(len(bin_betha2) - 8)]

        nl_bin_betha3 = bin_betha3[(len(bin_betha3) - 8):(len(bin_betha3))]
        nh_bin_betha3 = bin_betha3[(len(bin_betha3) - 12):(len(bin_betha3) - 8)]

        nl_dec_betha1 = int(nl_bin_betha1, 2)
        nh_dec_betha1 = int(nh_bin_betha1, 2)

        nl_dec_betha2 = int(nl_bin_betha2, 2)
        nh_dec_betha2 = int(nh_bin_betha2, 2)

        nl_dec_betha3 = int(nl_bin_betha3, 2)
        nh_dec_betha3 = int(nh_bin_betha3, 2)

        # print(nl_dec_betha2)
        # print(nh_dec_betha2)

        r1 = 1 + 5 + 3 + parametro + nl_dec_betha1 + nh_dec_betha1
        r2 = 2 + 5 + 3 + parametro + nl_dec_betha2 + nh_dec_betha2
        r3 = 3 + 5 + 3 + parametro + nl_dec_betha3 + nh_dec_betha3

        z1 = bin(r1)
        z1 = "0000000000" + z1[2:(len(z1))]
        z2 = bin(r2)
        z2 = "0000000000" + z2[2:(len(z2))]
        z3 = bin(r3)
        z3 = "0000000000" + z3[2:(len(z3))]

        z1 = z1[(len(z1) - 12):(len(z1))]
        z2 = z2[(len(z2) - 12):(len(z2))]
        z3 = z3[(len(z3) - 12):(len(z3))]
        # print(z1)

        not_z1 = bin(int(z1, 2) ^ 4095)
        not_z2 = bin(int(z2, 2) ^ 4095)
        not_z3 = bin(int(z3, 2) ^ 4095)
        #        print(not_z1)

        n11 = not_z1[(len(not_z1) - 8):(len(not_z1))]
        n22 = not_z2[(len(not_z2) - 8):(len(not_z2))]
        n33 = not_z3[(len(not_z3) - 8):(len(not_z3))]
        #        print(n11)

        p11 = int(n11, 2)  # este seria el checksum
        p22 = int(n22, 2)
        p33 = int(n33, 2)

        #        vector= [255,255,1,5,3,parametro,nl_dec_betha1,nh_dec_betha1,p11,255,255,2,5,3,parametro,nl_dec_betha2,nh_dec_betha2,p22,255,255,3,5,3,parametro,nl_dec_betha3,nh_dec_betha3,p33]
        vector_1 = [255, 255, 1, 5, 3, parametro, nl_dec_betha1, nh_dec_betha1, p11]
        vector_2 = [255, 255, 2, 5, 3, parametro, nl_dec_betha2, nh_dec_betha2, p22]
        vector_3 = [255, 255, 3, 5, 3, parametro, nl_dec_betha3, nh_dec_betha3, p33]

        ser.write(vector_1)
        ser.write(vector_2)
        ser.write(vector_3)

        mess_status = ser.read(1000)

        return mess_status

    # ------------------------------------------------------------------------------------------------

    # ===========================================================================
    def manipulador_mecanica(self):
        """Funcion principal
           Solo se debe ejecutar si se ejecuta el programa de forma individual
           Pero no se debe ejecutar si se carga como modulo dentro de otro programa
        """

        a = 80
        b = -100
        # Coordenadas de busqueda las marcas
        marcas_cad_mm = (
        [[150, -50], [310, -50], [130, 0], [330, 0], [170, 50], [290, 50], [100, 100], [360, 100], [250, 145],
         [435, 145], [80, 220], [380, 220], [170, 250], [290, 250], [80, 280], [380, 280], [130, 330], [300, 330],
         [230, 318.4]])
        marcas_cad_mm_1 = (
        [[150 + a, 650 + b], [130 + a, 600 + b], [330 + a, 600 + b], [170 + a, 550 + b], [290 + a, 550 + b],
         [100 + a, 500 + b], [360 + a, 500 + b], [80 + a, 380 + b], [380 + a, 380 + b], [170 + a, 350 + b],
         [80 + a, 320 + b], [380 + a, 320 + b], [130 + a, 270 + b], [230 + a, 281.6 + b]])
        marcas_cad_mm_neg = (
        [[0, 0], [150, 50], [310, 50], [130, 0], [330, 0], [170, -50], [290, -50], [100, -100], [360, -100], [25, -145],
         [435, -145], [80, -220], [380, -220], [170, -250], [290, -250], [80, -280], [380, -280], [130, -330],
         [300, -330], [230, -318.4]])
        marcas_click = (
        [[248, 428], [359, 431], [238, 389], [378, 392], [276, 354], [340, 355], [223, 322], [396, 323], [176, 288],
         [426, 291], [200, 234], [412, 246], [282, 219], [342, 224], [203, 189], [417, 198], [244, 180], [381, 184],
         [311, 173]])
        centroides_marcas = (
        [[247, 426], [237, 388], [376, 390], [276, 353], [339, 354], [222, 320.5], [395, 322], [199, 233], [411, 245],
         [280.5, 218], [202, 188], [416, 197], [242, 179], [310, 172]])

        referencia_marca_17_medido = [201, 195]
        referencia_marca_17_cad = [130, 330]
        nuevo_cero = [71, 525]

        # Se utiliza esta linea si se desea probar e manipulador con un video precargado
        #    cap = cv2.VideoCapture('videoFinalM.wmv')

        # Se utiliza esta linea si se desea probar directamente con la camara. Hay que especificar el numero de la camara en el sistema
        cap = cv2.VideoCapture(0)

        # Se captura el valor de la tasa de adquisicion del video para alimentar Kalman. Solo para video
        #    fps = cap.get(cv2.CAP_PROP_FPS)

        # Se especifica la tasa de captura para Kalman. ¿Como se hace?
        #    fps = 30

        # si no se conce la ubicacion de las marcas, se pueden indicar con la siguiente linea
        # marcas_click = prm.inicializar_marcas(img_base)

        # Se crea el objeto e la clase proyectividad
        estabilizador = ProyectividadOpenCV()

        # Se carga una imagen base para hallar la homografia de esta contra el CAD y asi utilizarla como base
        img_for_mm = cv2.imread("img_base.png")

        # Este metodo halla la homografia contra el cad desde una serie de puntos correspondientes a los centros estimados. El se encarga de buscar los centroides
        # img_base = estabilizador.estabilizar_desde_marcas(img_for_mm,marcas_click,marcas_cad_mm_1)

        # Este metodo halla la homografia contra el cad desde una serie de puntos correspondientes a los centroides de las marcas
        img_base = estabilizador.estabilizar_desde_centroides_marcas(img_for_mm, centroides_marcas, marcas_cad_mm_1)

        # Se crea una variable delta de t para kalman
        delta_t = 0.1

        # Se mide el tiempo que pasa entre la captura de un frame y otro
        tiempo_inicial = time()

        # Se inicializa Kalman
        cx, cy = 200, 200
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, delta_t, 0], [0, 1, 0, delta_t], [0, 0, 1, 0], [0, 0, 0, 1]],
                                           np.float32)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.01
        kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.0001
        tp = [23, 13]

        # Se abre el puerto serial y se deja abierto para una comunicacion constante
        ser = estabilizador.abrir_puerto_serial(puerto='COM6')  # Hay que verificar el puerto
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
        while (True):
            tiempo_final = time()
            delta_t = tiempo_final - tiempo_inicial
            tiempo_inicial = tiempo_final
            # Capture frame-by-frame
            ret, frame = cap.read()

            print(delta_t)

            # Esta clase estabiliza automaticamente la imagen con base una imagen inicial
            estabilizada = estabilizador.estabilizador_imagen(frame, img_base)

            # Se aplica un ruido gausiano para suavizar bordes
            blur = cv2.blur(estabilizada, (3, 3))
            # Se hace la transformacion a HSV
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            # Se aplica la mascara de color que solo deja pasar rojos. Video
            #        thresh_objeto = cv2.inRange(hsv,np.array((0,50,50)), np.array((10,255,255)))

            #        Se aplica la mascara de color que solo deja pasar rojos. Camara
            thresh_objeto = cv2.inRange(hsv, np.array((160, 100, 100)), np.array((179, 255, 255)))

            # se buscan los contornos en la imagen filtada para rojos
            image, contours, hierarchy = cv2.findContours(thresh_objeto, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Se inicializa la variable para contar los frames en los que se pierde el objeto para alimentar Kalman
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
                    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

                    # Se dibuja una cruz verde sobre el objeto encontrado
                    cv2.line(blur, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
                    cv2.line(blur, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)

                    # Se hace la conversion del formato para alimentar Kalman con el centroide
                    mp = np.array([[np.float32(cx)], [np.float32(cy)]])
                    tp = kalman.predict()
                    kalman.correct(mp)

                    # Como el objeto fue encontrado, entonces, se deja en 1 el conteo de frames
                    cont_frame = 1

                # Si no seencuentra el objeto, se hace la estimacion de Kalman y no se actualiza la medida
                if area <= max_area or area >= 750:
                    kalman.transitionMatrix = np.array(
                        [[1, 0, cont_frame * delta_t, 0], [0, 1, 0, cont_frame * delta_t], [0, 0, 1, 0], [0, 0, 0, 1]],
                        np.float32)

                    # Se hace el conteo de los frames en los que no se encuentra la marca para alimentar Kalman
                    cont_frame = cont_frame + 1

                    # Sedibuja una cruz azul en la posicion estimada del objeto
                    tp = kalman.predict()
                    cv2.line(blur, (tp[0] - 10, tp[1]), (tp[0] + 10, tp[1]), (255, 0, 0), 1)
                    cv2.line(blur, (tp[0], tp[1] - 10), (tp[0], tp[1] + 10), (255, 0, 0), 1)

            # Se halla la varianza de la estimacion para reducir la zona de busqueda en el siguiente frame
            varianza_x = kalman.errorCovPost[0, 0]
            varianza_y = kalman.errorCovPost[1, 1]
            devstd_x = varianza_x ** 0.5
            devstd_y = varianza_y ** 0.5

            # para 6 desviaciones estandar
            marco_x = devstd_x * 6
            marco_y = devstd_y * 6

            # Se dibuja un circulo blanco que crece con la varianza a razon de 6 desviaciones estandar
            cv2.circle(blur, (int(tp[0][0]), int(tp[1][0])), int(marco_x), (255, 255, 255), 1, cv2.LINE_AA)
            #        print(tp)

            # Se obtienen las coordenadas en centimetros para la cinematicainversa
            cor_x = 10 + (tp[0][0] - nuevo_cero[0]) / 10
            cor_y = (nuevo_cero[1] - tp[1][0]) / 10
            #        print(cor_x,cor_y)
            if cor_x <= 14 or cor_x >= 34:
                cor_x = 23
            if cor_y <= 4 or cor_y >= 26:
                cor_y = 13.28

            #        print(cor_x,cor_y)
            # Se utiliza la cinematica inversa para obtener el valr de los angulos en decimal de 0 a 4096
            b1, b2, b3 = estabilizador.cinematica_inversa(cor_x, cor_y)
            ##        print(tp[0]/10,tp[1]/10)
            #        print(angulos_decimales)

            if ser.isOpen():
                mensaje_status = estabilizador.enviar_tethas_servomotores(ser, b1, b2, b3)

            #        print(mensaje_status)
            # Visualizacion de imagenes
            cv2.imshow('Umbral', blur)
            cv2.imshow('Mask', thresh_objeto)
            #    cv2.imshow('Marcas',thresh_marcas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        ser.close()
        cap.release()
        cv2.destroyAllWindows()

    def downsample_image(self, image, reduce_factor):
        for i in range(0, reduce_factor):
            # Check if image is color or grayscale
            if len(image.shape) > 2:
                row, col = image.shape[:2]
            else:
                row, col = image.shape

            image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
        return image

    # Function to create point cloud file
    def create_output(self, vertices, colors, filename):
        colors = colors.reshape(-1, 3)
        vertices = np.hstack([vertices.reshape(-1, 3), colors])

        ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		'''
        with open(filename, 'w') as f:
            f.write(ply_header % dict(vert_num=len(vertices)))
            np.savetxt(f, vertices, '%f %f %f %d %d %d')

    def disparity_map(self, img_0, img_1):
        "Esta clase calcula el mapa de disparidad de dos imagenes y lo entrega como una nueva imagen visualizable"
        "Recibe las imágenes y la ruta de los parḿetros de cámara"
        "Entrega la imagen del mapa de disparidad"

        # Load camera parameters
        ret_0 = np.load('./ejemplos/example_9/cam_0_params/ret.npy')
        K_0 = np.load('./ejemplos/example_9/cam_0_params/K.npy')
        dist_0 = np.load('./ejemplos/example_9/cam_0_params/dist.npy')

        ret_1 = np.load('./ejemplos/example_9/cam_1_params/ret.npy')
        K_1 = np.load('./ejemplos/example_9/cam_1_params/K.npy')
        dist_1 = np.load('./ejemplos/example_9/cam_1_params/dist.npy')

        # Specify image paths
        #img_path1 = './ejemplos/left2.jpg'
        #img_path2 = './ejemplos/right2.jpg'

        # Load pictures
        #img_0 = cv2.imread(img_path1)
        #img_1 = cv2.imread(img_path2)

        # Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size
        h, w = img_1.shape[:2]
        # print(h, w)

        # Get optimal camera matrix for better undistortion
        new_camera_matrix_0, roi_0 = cv2.getOptimalNewCameraMatrix(K_0, dist_0, (w, h), 1, (w, h))
        new_camera_matrix_1, roi_1 = cv2.getOptimalNewCameraMatrix(K_1, dist_1, (w, h), 1, (w, h))

        # Undistort images
        img_0_undistorted = cv2.undistort(img_0, K_0, dist_0, None, new_camera_matrix_0)
        img_1_undistorted = cv2.undistort(img_1, K_1, dist_1, None, new_camera_matrix_1)

        # Downsample each image 3 times (because they're too big)
        img_0_downsampled = self.downsample_image(img_0_undistorted, 0)
        img_1_downsampled = self.downsample_image(img_1_undistorted, 0)

        # Set disparity parameters
        # Note: disparity range is tuned according to specific parameters obtained through trial and error.
        win_size = 5
        min_disp = -1
        max_disp = 63  # min_disp * 9
        num_disp = max_disp - min_disp  # Needs to be divisible by 16

        # Create Block matching object.
        stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                       numDisparities=num_disp,
                                       blockSize=5,
                                       uniquenessRatio=5,
                                       speckleWindowSize=5,
                                       speckleRange=5,
                                       disp12MaxDiff=2,
                                       P1=8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                                       P2=32 * 3 * win_size ** 2)  # 32*3*win_size**2)

        # Compute disparity map
        #print("\nComputing the disparity  map...")
        disparity_map = stereo.compute(img_0_downsampled, img_1_downsampled)
        img_disparity_map = self.matrix_to_image(disparity_map)

        return img_disparity_map

    def stereo_3D_reconstruction(self):
        "Esta clase realiza la reconstrucción de una escena 3D con base en un conjunto de imágene de la misma"
        "Recibe la URL dnde están guardados los parámetros de cámara y la URL de las imágenes"
        "Entrega la escena reconstruida"


        # Load camera parameters
        ret_0 = np.load('./ejemplos/example_9/cam_0_params/ret.npy')
        K_0 = np.load('./ejemplos/example_9/cam_0_params/K.npy')
        dist_0 = np.load('./ejemplos/example_9/cam_0_params/dist.npy')

        ret_1 = np.load('./ejemplos/example_9/cam_1_params/ret.npy')
        K_1 = np.load('./ejemplos/example_9/cam_1_params/K.npy')
        dist_1 = np.load('./ejemplos/example_9/cam_1_params/dist.npy')

        # Specify image paths
        img_path1 = './ejemplos/nube2/image_0.jpg'
        img_path2 = './ejemplos/nube1/image_0.jpg'

        # Load pictures
        img_0 = cv2.imread(img_path1)
        img_1 = cv2.imread(img_path2)

        # Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size
        h, w = img_1.shape[:2]
        #print(h, w)

        # Get optimal camera matrix for better undistortion
        new_camera_matrix_0, roi_0 = cv2.getOptimalNewCameraMatrix(K_0, dist_0, (w, h), 1, (w, h))
        new_camera_matrix_1, roi_1 = cv2.getOptimalNewCameraMatrix(K_1, dist_1, (w, h), 1, (w, h))

        # Undistort images
        img_0_undistorted = cv2.undistort(img_0, K_0, dist_0, None, new_camera_matrix_0)
        img_1_undistorted = cv2.undistort(img_1, K_1, dist_1, None, new_camera_matrix_1)

        # Downsample each image 3 times (because they're too big)
        img_0_downsampled = self.downsample_image(img_0_undistorted, 0)
        img_1_downsampled = self.downsample_image(img_1_undistorted, 0)

        # Set disparity parameters
        # Note: disparity range is tuned according to specific parameters obtained through trial and error.
        win_size = 1
        min_disp = 0
        max_disp = 16  # min_disp * 9
        num_disp = max_disp - min_disp  # Needs to be divisible by 16

        # Create Block matching object.
        stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                       numDisparities=num_disp,
                                       blockSize=2,
                                       uniquenessRatio=5,
                                       speckleWindowSize=5,
                                       speckleRange=5,
                                       disp12MaxDiff=2,
                                       P1=8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                                       P2=32 * 3 * win_size ** 2)  # 32*3*win_size**2)

        # Compute disparity map
        print("\nComputing the disparity  map...")
        disparity_map = stereo.compute(img_0_downsampled, img_1_downsampled)
        img_disparity_map = self.matrix_to_image(disparity_map)

        print("Cierre la ventana para continuar")

        # Show disparity map before generating 3D cloud to verify that point cloud will be usable.
        cv2.imshow('Disparity Map', img_disparity_map)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        #plt.imshow(disparity_map, 'gray')
        #plt.savefig("disparity.png")
        #plt.show()

        # Generate  point cloud.
        print("\nGenerating the 3D map...")

        # Get new downsampled width and height
        h, w = img_1_downsampled.shape[:2]

        focal_length = (K_0[0,0]+K_0[1,1]+K_1[0,0]+K_1[1,1])/4

        # Load focal length.
        #focal_length = np.load('./camera_params/FocalLength.npy')

        # Perspective transformation matrix
        # This transformation matrix is from the openCV documentation, didn't seem to work for me.
        Q = np.float32([[1, 0, 0, -w / 2.0],
                        [0, -1, 0, h / 2.0],
                        [0, 0, 0, -focal_length],
                        [0, 0, 1, 0]])

        # This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision.
        # Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
        Q2 = np.float32([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, focal_length * 0.0001, 0],  # Focal length multiplication obtained experimentally.
                         [0, 0, 0, 1]])

        # Reproject points into 3D
        points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
        # Get color points
        colors = cv2.cvtColor(img_1_downsampled, cv2.COLOR_BGR2RGB)

        # Get rid of points with value 0 (i.e no depth)
        mask_map = disparity_map > disparity_map.min()
        #mask_map = disparity_map

        # Mask colors and points.
        output_points = points_3D[mask_map]
        output_colors = colors[mask_map]

        # Define name for output file
        output_file = 'reconstructed.ply'

        # Generate point cloud
        print("\n Creating the output file... \n")
        self.create_output(output_points, output_colors, output_file)

    def video_from_camera(self, num_camera=0):
        cap = cv2.VideoCapture(num_camera)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # print(fps)
        # fgbg = cv2.createBackgroundSubtractorMOG2(5, 10, True)

        while (True):

            # Capture frame-by-frame
            ret, frame = cap.read()

            #    cv2.imshow('ndvi4',ndvi4)
            cv2.imshow('original', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        return frame

    def focal_length_from_image(self, path_image):
        "Esta clase extrae la distancia focal de la información EXIF de una imagen"

        # Get exif data in order to get focal length.
        exif_img = PIL.Image.open(path_image)
        # print(exif_img)
        exif_data = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in exif_img._getexif().items()
            if k in PIL.ExifTags.TAGS}
        # Get focal length in tuple form
        focal_length_exif = exif_data['FocalLength']
        # print(focal_length_exif)
        # Get focal length in decimal form
        focal_length = focal_length_exif[0] / focal_length_exif[1]
        # np.save("./camera_params/FocalLength", focal_length)
        return focal_length

    def video_stabilizer_full(self, video_input="ejemplos/example_1/billarVideo.mp4"):

        cap = cv2.VideoCapture(video_input)

        img_base = cv2.imread("ejemplos/example_1/baseBillar.png")
        img_base = cv2.resize(img_base, (900, 500), interpolation=cv2.INTER_LINEAR)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (900, 500))

        # estabilizador = self.ProyectividadOpenCV()

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = cv2.resize(frame, (900, 500), interpolation=cv2.INTER_LINEAR)
            estabilizada = self.estabilizador_imagen(frame, img_base)
            out.write(estabilizada)

            # Display the resulting frame
            cv2.imshow('frame', estabilizada)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        out.release()
        return "Video is ready"
        cap.release()
        cv2.destroyAllWindows()

    def ndvi_calculation(self, url_img_RED="ejemplos/example_2/img_RED.TIF",
                         url_img_NIR="ejemplos/example_2/img_NIR.TIF", width=700, height=500):
        "En esta clase se calcula el índice NDVI a partir de un par de imágenes entregadas en el argumento"

        "Se leen las imágenes"
        img_RED = cv2.imread(url_img_RED, 0)
        stb_NIR = cv2.imread(url_img_NIR, 0)

        img_RED = cv2.resize(img_RED, (width, height), interpolation=cv2.INTER_LINEAR)
        stb_NIR = cv2.resize(stb_NIR, (width, height), interpolation=cv2.INTER_LINEAR)

        "Se alinean as imágenes y se utiliza la imagen roja como imagen base"

        stb_RED = self.estabilizador_imagen(img_RED, stb_NIR)

        "Se convierten las imágenes en arreglos trabajables con numpy y matplotlib"
        red = array(stb_RED, dtype=float)
        nir = array(stb_NIR, dtype=float)

        "Se verifican y corrigen las divisiones por cero"
        check = np.logical_and(red > 1, nir > 1)

        "Se calcula el índice ndvi"
        ndvi = np.where(check, (nir - red) / (nir + red), 0)

        ndvi_index = ndvi

        "Se verifica que todos los valores queden sobre cero"
        if ndvi.min() < 0:
            ndvi = ndvi + (ndvi.min() * -1)

        ndvi = (ndvi * 255) / ndvi.max()
        ndvi = ndvi.round()

        ndvi_image = np.array(ndvi, dtype=np.uint8)

        return ndvi_index, ndvi_image

    def image_collection_for_calibration(self, num_camera=1, num_images=15, url_collection="./ejemplos/example_9/cam_"):
        "Esta clase permite construir un directorio de imágenes de cantidad variable desde una cámara"
        "Recibe numero camara DEF(1), numero de imagenes DEF(15) y URL para guardar imagenes DEF(example_9)"

        "Captura de conjunto de imagenes para calibracion"
        image_list = []
        url_collection = url_collection + str(num_camera) + "/"
        # Si el directorio no existe, se crea
        try:
            os.stat(url_collection)
        except:
            os.mkdir(url_collection)

        for i in range(num_images):
            frame = self.video_from_camera(num_camera)
            # Crea una carpeta para cada camara
            label_image = url_collection + "image_" + str(i) + ".jpg"
            cv2.imwrite(label_image, frame)
            image_list.append(frame)
            print(str(i) + " image saved from camera " + str(num_camera))
        print("Image collection completed")
        return

    def cam_calibration_with_img_collection(self, url_img_collection="./ejemplos/example_9/cam_",
                                            chessboard_size=(9, 6), num_camera=1):
        "Clase para calibracion de camara con colección de imágenes disponible"
        "Recibe el tamaño del tablero y la ruta para la colección de imágenes"
        print("Ingreso a la clase")

        url_img_collection = url_img_collection + str(num_camera) + "/*"
        # chessboard_size = (9, 6)

        # Define arrays to save detected points
        obj_points = []  # 3D points in real world space
        img_points = []  # 3D points in image plane
        # Prepare grid and points to display
        objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        # read images
        # calibration_paths = glob.glob('/ejemplos/example_9/image_?.jpg')
        calibration_paths = glob.glob(url_img_collection)
        print(len(calibration_paths))
        # Iterate over images to find intrinsic matrix
        for image_path in tqdm(calibration_paths):
            # Load image
            image = cv2.imread(image_path)
            # print(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("Image loaded, Analizying...")
            # find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
            if ret == True:
                print("Chessboard detected!")
                print(image_path)
                # define criteria for subpixel accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                # refine corner location (to subpixel accuracy) based on criteria.
                cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners)

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_image.shape[::-1], None, None)

        # Save parameters into numpy file
        url_parametros = "./ejemplos/example_9/cam_" + str(num_camera) + "_params/"

        #Si el directorio no existe, se crea
        try:
            os.stat(url_parametros)
        except:
            os.mkdir(url_parametros)

        #Se guardan los datos de calibración como archivos np para que estén disponibles
        np.save(url_parametros+"ret", ret)
        np.save(url_parametros+"K", K)
        np.save(url_parametros+"dist", dist)
        np.save(url_parametros+"rvecs", rvecs)
        np.save(url_parametros+"tvecs", tvecs)

        return ret, K, dist, rvecs, tvecs

    def cam_calibration_without_img_collection(self, chessboard_size=(9, 6), num_images=15, num_camera=1,
                                               url_collection="./ejemplos/example_9/cam_"):
        "Esta clase permite crear la colección de imágenes y hacer la calibración de cámara"
        "Recibe el tamaño del tablero DEF(9X6), la cantidad de imágenes para la colecciónDEF(15) y la cámara a calibrar DEF(1)"
        "Retorna los cinco parámetros de calibración de camara"

        # Creando la collección de imágenes
        print("Paso 1: Creando la collección de imágenes \n")
        print("Siga las instrucciones paso a paso")
        self.image_collection_for_calibration(num_camera, num_images, url_collection)

        # Calibrando la cámara
        print("Paso 2: Calculando parámetros de calibración \n")
        print("Este proceso puede tardar varios segundos")
        ret, K, dist, rvecs, tvecs = self.cam_calibration_with_img_collection(url_collection, chessboard_size,
                                                                              num_camera)

        return ret, K, dist, rvecs, tvecs

    def matrix_to_image(self, mtx):
        "Esta clase toma un arreglo  en cualquier formato y lo convierte en una imagen para openCV"

        "Se verifica que todos los valores queden sobre cero"
        if mtx.min() < 0:
            mtx = mtx + (mtx.min() * -1)

        mtx = (mtx * 255) / mtx.max()
        mtx = mtx.round()

        final_img = np.array(mtx, dtype=np.uint8)
        return final_img

    def point_cloud_generation(self, focal_length, img_1_downsampled, disparity_map):

        "Este método permite generar una nube de punrtos y visualizarla con Open3D"

        # Generate  point cloud.
        print("\nGenerating the 3D map...")

        # Get new downsampled width and height
        h, w = img_1_downsampled.shape[:2]

        #focal_length = (K_0[0, 0] + K_0[1, 1] + K_1[0, 0] + K_1[1, 1]) / 4

        # Load focal length.
        # focal_length = np.load('./camera_params/FocalLength.npy')

        # Perspective transformation matrix
        # This transformation matrix is from the openCV documentation, didn't seem to work for me.
        Q = np.float32([[1, 0, 0, -w / 2.0],
                        [0, -1, 0, h / 2.0],
                        [0, 0, 0, -focal_length],
                        [0, 0, 1, 0]])

        # This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision.
        # Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
        Q2 = np.float32([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, focal_length * 0.0001, 0],  # Focal length multiplication obtained experimentally.
                         [0, 0, 0, 1]])

        # Reproject points into 3D
        points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
        # Get color points
        colors = cv2.cvtColor(img_1_downsampled, cv2.COLOR_BGR2RGB)

        # Get rid of points with value 0 (i.e no depth)
        mask_map = disparity_map > disparity_map.min()
        # mask_map = disparity_map

        # Mask colors and points.
        output_points = points_3D[mask_map]
        output_colors = colors[mask_map]

        # Define name for output file
        output_file = 'reconstructed.ply'

        # Generate point cloud
        print("\n Creating the output file... \n")
        self.create_output(output_points, output_colors, output_file)


def main():
    print("\n Computer Vision Class by Jorge Martínez \n \n")
    print("Type 1 for Video Stabilizer example \n")
    print("Type 2 for Photogrametric Alignment example \n")
    print("Type 3 for NDVI example \n")

    ejemplo = input()

    #Ejemplo estable para estabilizador de video or software
    if ejemplo == "1":
        "Ejemplo estable para hacer estabilización de video"
        print("Descargue el video billarVideo.mp4, ubíquelo en la carpeta ejemplos/example_1 y oprima enter\n")
        print("\nType Ok to verify")
        url_video_input = input()

        example_1 = ProyectividadOpenCV()
        print("Este proceso puede tardar varios minutos dependiendo del tamaño del video")
        print("El tamaño al que se redimencioann todos los videos por defecto es 700X500 \n")
        video_stabilizer = example_1.video_stabilizer_full()
        print("Trabajo finalizado, el video estabilizado puede encontrarse en la carpeta VisionPC junto a la clase")

    #Ejemplo estable para alineación fotogramétrica de las imágenes Sequoia
    elif ejemplo == "2":
        "Ejemplo estable para hacer alineación fotogramétrica de imágenes"
        print("Comenzado proceso con imágenes de prueba propias \n")
        print(
            "Por favor ingrese ancho y alto deseado para las imágenes \npor ejemplo 700,500. De no asignarlos, los valores por defecto serán 700X500 \n \n")
        print("Ancho=")
        width_str = input()
        print("Alto=")
        height_str = input()

        width = int(width_str)
        height = int(height_str)

        example_2 = ProyectividadOpenCV()

        img_RGB = cv2.imread("ejemplos/example_2/img_RGB.JPG", 0)
        img_GRE = cv2.imread("ejemplos/example_2/img_GRE.TIF", 0)
        img_NIR = cv2.imread("ejemplos/example_2/img_NIR.TIF", 0)
        img_RED = cv2.imread("ejemplos/example_2/img_RED.TIF", 0)
        img_REG = cv2.imread("ejemplos/example_2/img_REG.TIF", 0)

        merged_fix_bad = cv2.merge((img_GRE, img_RED, img_NIR))
        merged_fix_bad = cv2.resize(merged_fix_bad, (width, height), interpolation=cv2.INTER_LINEAR)

        stb_RGB, stb_GRE, stb_NIR, stb_RED, stb_REG = example_2.img_alignment_sequoia(img_RGB, img_GRE, img_NIR,
                                                                                      img_RED, img_REG, width, height)

        merged_fix_stb = cv2.merge((stb_GRE, stb_RED, stb_NIR))

        print(
            "La primera imagen que se genera simplemente superpone las imágenes sin alinear \n Cerrar la ventana para continuar \n")
        cv2.imshow('frame', merged_fix_bad)
        cv2.waitKey(0)

        print("La siguiente imagen si se encuentra debidamente alineada. Cerrar la ventana para terminar")
        cv2.imshow('frame', merged_fix_stb)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    #Ejemplo estable para generar imagen NDVI con NIR y RED
    elif ejemplo == "3":
        "Ejemplo estable para generar imagen NDVI"
        print("Se construye el objeto")
        example_3 = ProyectividadOpenCV()

        print("Para este ejemplo se utilizará el mismo conjunto de imágenes del ejemplo 2")
        url_img_RED = "ejemplos/example_2/img_RED.TIF"
        url_img_NIR = "ejemplos/example_2/img_NIR.TIF"

        "Se envían las URL y se obtienen los índices NDVI y una imagen adecuada para visualizar"

        ndvi_index, ndvi_image = example_3.ndvi_calculation(url_img_RED, url_img_NIR)

        "Se pinta la imagen con colormap de OpenCV. En mi caso, RAINBOW fue la mejor opción"
        im_color = cv2.applyColorMap(ndvi_image, cv2.COLORMAP_RAINBOW)

        print(ndvi_index)

        cv2.imshow('frame', im_color)
        cv2.waitKey(0)

        print("Cerrar la ventana para finalizar")

    #Ejemplo de prueba para abrir cámaras del pc (Se ingresa el número de cámara a visualizar)
    elif ejemplo == "4":
        "Ejemplo de prueba para abrir cámaras en el PC"

        print("Digite el numero de camara que desea visualizar")
        num_camera = input()

        example_4 = ProyectividadOpenCV()
        example_4.video_from_camera(int(num_camera))

    #Espacio libre
    elif ejemplo == '5':
        "Ejemplo de prueba"
        print("ejemplo 3D")

        imgL = cv2.imread('ejemplos/l_img_0.jpg', 0)
        imgR = cv2.imread('ejemplos/r_img_0.jpg', 0)

        stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL, imgR)
        plt.imshow(disparity, 'gray')
        plt.show()

    #Ejemplo estable para reconstrucción de imagen 3D con cámaras calibradas e imágenes guardadas previamente
    elif ejemplo == '6':
        "Ejemplo estable para reconstrución estereoscópica"
        example_6 = ProyectividadOpenCV()
        example_6.stereo_3D_reconstruction()

        cloud = o3d.io.read_point_cloud("reconstructed.ply")  # Read the point cloud
        o3d.visualization.draw_geometries([cloud])  # Visualize the point cloud

    #Ejemplo estable para visualizar el mapa de disparidad con dos cámaras en tiemo real (estar seguro del número de la cámara)
    elif ejemplo == '7':
        "Ejemplo estable para calcular y visualiar mapa de disparidad en tiempo real"

        example_7 = ProyectividadOpenCV()

        cap_0 = cv2.VideoCapture(1)
        cap_1 = cv2.VideoCapture(2)
        fps = cap_0.get(cv2.CAP_PROP_FPS)

        while (True):

            # Capture frame-by-frame
            ret_0, frame_0 = cap_0.read()
            ret_1, frame_1 = cap_1.read()

            disparity_map = example_7.disparity_map(frame_0, frame_1)

            cv2.imshow('Disparity Map', disparity_map)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap_0.release()
        cap_1.release()
        cv2.destroyAllWindows()

    #Ejemplo estable para realizar la calibración de cámara teniendo la colección de imágenes previamente guardadas
    elif ejemplo == "8":
        "Ejemplo estable para Calibracion de camara"
        print("Camera calibration")

        example_8 = ProyectividadOpenCV()
        example_8.cam_calibration_with_img_collection()
        print("Calibración terminada")

    #Ejemplo estable para crear colecciones imágenes y guardarlas en alguna carpeta del pc (para calibración de cámara)
    elif ejemplo == '9':

        "Ejemplo estable permite crear una coleccion de imágenes y guardarlas en la carpeta"

        example_9 = ProyectividadOpenCV()
        print("Type image quantity what you want:\n")
        image_quantity = input()
        print("\nType num camera")
        num_camera=input()
        url_collection = "./ejemplos/nube"
        example_9.image_collection_for_calibration(int(num_camera), int(image_quantity), url_collection)

    else:
        print("You typed:")
        print(ejemplo)
        print("And this is not a valid option")


# ===========================================================================
if __name__ == '__main__':
    main()
