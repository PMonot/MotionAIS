import os
import sys
if sys.__stdout__ is None or sys.__stderr__ is None:
    os.environ['KIVY_NO_CONSOLELOG'] = '1'

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.config import Config
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.graphics import Color, Line, Ellipse
from kivy.uix.label import Label
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg 

import json
import cv2
import csv
import math
import time
import copy
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from scipy.ndimage import gaussian_filter1d, median_filter


import read_raw_file as RRF
import marker_detection_with_particles


class MyApp(Widget):
    Window.maximize()
    Builder.load_file('design_interface.kv')
    Config.set('graphics', 'width', '1920')
    Config.set('graphics', 'height', '1080')

    global path
    path = ''

    # Fonction pour que les boutons changent de couleur lorsqu'enfoncés
    def press_color(self):
        self.background_normal = ''
        self.background_color = (100/255.0, 197/255.0, 209/255.0, 1)

    # Fonction pour sélectionner le répertoire .raw, puis définir le nombre d'images
    def im_select(self):
        print(self.width, self.height)
        timer_debut_im = time.process_time_ns()
        
        # flag pour savoir quelles infos sont disponibles
        global detection_eff
        detection_eff = False
        global coordo_xyz
        coordo_xyz = False
        global labelize_extent
        labelize_extent = False

        # Initie la variable pour numéro de l'image affichée à 1 pour voir la première image
        global image_nb
        image_nb = 1

        # chemins général/spécifiques vers les données de base/créées
        global path
        path = self.ids.path_input.text
        save_path = path+'/intensity/'
        global save_path_xyz
        save_path_xyz = path+'/xyz_images/'
        global save_path_im
        save_path_im = path+'/Preprocessed/'

        try: # si chemin entré valide
            # Crée les répertoires pour images converties et prétraitées
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(save_path_xyz, exist_ok=True)
            os.makedirs(save_path_im, exist_ok=True)

            # lit les fichiers .raw si pas déjà fait et crée les images
            if len(os.listdir(save_path)) == 0:
                RRF.read_raw_intensity_frames(path, save_path)
            # enregistre les fichiers _XYZ_.raw en png pour utilisation future des coordos xyz
            if len(os.listdir(save_path_xyz)) == 0:
                RRF.read_raw_xyz_frames(path)
            # définit les dimensions pour rogner les images
            self.automatic_crop()

            # crée les images Preprocessed pour consultation
            global im_dim
            if len(os.listdir(save_path_im)) == 0:
                for filename_i, filename_xyz in zip(os.listdir(save_path), os.listdir(save_path_xyz)):
                    frame_display, preprocessed_frame = marker_detection_with_particles.preprocess(cv2.imread(os.path.join(save_path, filename_i)), self.remove_bg(np.load(os.path.join(save_path_xyz, filename_xyz))), w1, w2, h1, h2)
                    cv2.imwrite(os.path.join(save_path_im, filename_i), preprocessed_frame)
                im_dim = preprocessed_frame.shape
            else:
                im_dim = cv2.imread(os.path.join(save_path_im, os.listdir(save_path_im)[0])).shape

            # Trouve le nombre d'images, définit le max du slider et le texte /tot
            global images_total
            images_total = len(os.listdir(save_path))
            self.ids.slider.max = images_total
            self.ids.image_total.text = f'/{images_total}'
            self.ids.label_ready.text = "Images prêtes, bougez le curseur ou entrez un numéro d'image"
        
            # Initie les variables pour dictionnaire de coordonnées et flag analyse_eff (pour affichage x,y,z)
            global dict_coordo
            dict_coordo = {}
            for image_id in range(len(os.listdir(save_path_im))):
                dict_coordo[f"image{image_id+1}"] = []
            global dict_coordo_labels_manual
            dict_coordo_labels_manual = {}

            global nb_marqueurs
            nb_marqueurs = np.nan
            
            # affiche la 1re image
            self.show_image()

            timer_fin_im = time.process_time_ns()
            print(timer_debut_im, timer_fin_im)
            print(f'Temps création images + détection marqueurs : {timer_fin_im - timer_debut_im} ns')

        except FileNotFoundError:
            self.ids.label_ready.text = "Le chemin entré est introuvable. Essayez à nouveau."

        #Si mode ouvrir: récupérer les positions existantes
        global labels
        if self.ids.check_open.active:
            pos_path = os.path.join(path,'Positions')
            if os.path.exists(pos_path) and len(os.listdir(pos_path)) > 0:
                with open(os.path.join(path,'Positions','positions_corrigees.json'), 'r') as positions:
                    dict_coordo_labels_manual = json.load(positions)

                for key, dict_value in dict_coordo_labels_manual.items():
                    dict_coordo[key] = list(dict_value.values())
                    if len(dict_value.keys()) > 0:
                        labels = list(dict_value.keys()) 

                if labels:
                    nb_of_marqueurs = len(labels)
                    self.change_grid_nb_marqueurs(nb_of_marqueurs)

                    labelize_extent = True
                    detection_eff = True
        

    # Prend une image xyz et retourne z en binaire 
    def remove_bg(self, xyz):
        z = xyz[:,:,2]

        zz = z[np.where(z>0)]
        zz = zz[np.where(zz<2500)]
        z_nobg = copy.deepcopy(z)
        body_z = np.quantile(zz, 0.3)
        if 'Contraint' in os.listdir(save_path_xyz)[0]:
            z_nobg[np.where(z > body_z + 150)] = False
        else:
            z_nobg[np.where(z > body_z + 250)] = False
        z_nobg = median_filter(z_nobg, 3)

        return z_nobg

    # Trouve les paramètres pour rogner les images (utilisé pour preprocess (RRF) et marker_detection)
    def automatic_crop(self):
        timer_debut = time.process_time_ns()

        xyz = np.load(os.path.join(save_path_xyz, os.listdir(save_path_xyz)[0]))
        z_nobg = self.remove_bg(xyz)
        body_LR = np.argwhere(z_nobg[1250,:]) #identifie points n'appartenant pas au bg, donc au corps du patient
        body_HL = np.argwhere(z_nobg[:,600])

        left = int(body_LR[0,0])
        right = int(body_LR[-1,0])

        global w1
        global w2
        global h1
        global h2

        if 'BG' in os.listdir(save_path_xyz)[0]:
            print('BG')
            w1 = np.max(left-100, 0)
            w2 = right+50
            h1 = int(body_HL[0,0])+100
        elif 'BD' in os.listdir(save_path_xyz)[0]:
            print('BD')
            w1 = left-50
            w2 = right+100
            h1 = int(body_HL[0,0])+100
        else:
            print('other')
            w1 = np.max(left-50, 0)
            w2 = right+50
            h1 = int(body_HL[0,0])-100

        h2 = h1+int(6/5*(w2-w1))+150
        print(w1, w2, h1, h2)

        self.ids.width.text = f'({w2-w1}, 0)'
        self.ids.height.text = f'(0, {h2-h1})'

        timer_fin = time.process_time_ns()
        print(timer_debut, timer_fin)
        print(f'Temps automatic crop : {timer_fin - timer_debut} ns')


    def begin_labelization(self):
        if self.ids.marq_nb_input.text == '':
            try:
                nb_of_marqueurs = len(dict_coordo[f"image{image_nb}"])
                if nb_of_marqueurs == 0:
                    warnings.warn("Impossible de commencer la labelization sans au moins un marqueur sur l'image")
                    pass
                else:
                    self.change_grid_nb_marqueurs(nb_of_marqueurs)
            except KeyError as e:
                print(e)
                pass

    # Fonction pour définir le nombre de marqueurs utilisés
    def nb_marqueurs_input(self):
        nb_of_marqueurs = int(self.ids.marq_nb_input.text)
        self.change_grid_nb_marqueurs(nb_of_marqueurs)
    
    def change_grid_nb_marqueurs(self,nb_of_marqueurs):
        global nb_marqueurs
        nb_marqueurs = nb_of_marqueurs
        self.ids.grid.size_hint = (.22, .04 + .02*nb_marqueurs) # taille du tableau variable selon le nombre de marqueurs
        self.ids.grid.rows = 1 + nb_marqueurs
        print(f'{nb_marqueurs} marqueurs utilisés')
        self.ids.marq_nb_input.text = str(nb_marqueurs)
        self.ids.nb_marqueurs.color = (1,1,1,1)

    # Fonction pour sélectionner le numéro de l'image à afficher, actualise la position du curseur
    def image_nb_input(self):
        global image_nb
        image_nb = int(self.ids.image_nb_input.text) # Définit numéro image actuelle pour détection marqueurs
        if 0 < image_nb <= images_total:
            self.ids.slider.value = image_nb #lien entre position du curseur et numéro de l'image
            self.show_image()
            self.canvas.remove_group(u"new_mark") # efface les points des marqueurs ajoutés
        else:
            self.ids.image_nb_input.text = 'Invalide' #si numéro entré à l'extérieur de l'intervalle acceptable

    # Fonction pour parcourir les images avec le curseur, actualise le numéro de l'image
    def slider_pos(self, *args):
        global image_nb
        image_nb = args[1] # définit numéro image actuelle pour détection marqueurs
        if 0 < image_nb  <= images_total:
            self.show_image()
            self.canvas.remove_group(u"new_mark") # efface les points des marqueurs ajoutés

    # Fonction pour afficher l'image et effacer les informations relatives à la précédente (marqueurs)        
    def show_image(self):
        self.ids.image_nb_input.text = f'{image_nb}'
        self.ids.image_show.clear_widgets()

        self.ids.image_show.source = os.path.join(save_path_im, sorted(os.listdir(save_path_im))[image_nb-1])

        self.canvas.remove_group(u"circle") # efface les cercles verts des marqueurs
        self.ids.rep_continuity.text = ''
        
        # Affiche les marqueurs si bouton activé
        if self.ids.button_showmarks.state == 'down':
            self.show_marqueurs()
            
        # Affiche les numéros d'images n'ayant pas le bon nb de marqueurs si bouton activé
        if self.ids.button_verif_nb.state == 'down':
            self.verif_nb()
        # Tag marqueurs discontinus si présents dans l'image actuelle
        if self.ids.button_verif_continuity.state == 'down':
            im_prob_continuity = self.verif_continuity()
            for el in im_prob_continuity:
                if image_nb == el:
                    self.ids.rep_continuity.text = 'Marqueur(s) à vérifier :\n'
                    for m in im_prob_continuity[image_nb]:
                        self.ids.rep_continuity.text += f'{m}, '
                    self.ids.rep_continuity.text = self.ids.rep_continuity.text[:-2]

        # Affiche positions dans le tableau après labellisation manuelle
        if labelize_extent and not coordo_xyz:
            self.ids.grid.clear_widgets()
            self.ids.grid.add_widget(Label(text='Marqueurs', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text='Positions', color=(0,0,0,1)))
            for l in labels:
                self.ids.grid.add_widget(Label(text=f'{l}', color=(0,0,0,1)))
                if l in dict_coordo_labels_manual[f'image{image_nb}']:
                    p = dict_coordo_labels_manual[f'image{image_nb}'][l]
                    self.ids.grid.add_widget(Label(text=f'({p[0]:.0f}, {p[1]:.0f})', color=(0,0,0,1)))
                else:
                    self.ids.grid.add_widget(Label(text=f'?', color=(0,0,0,1)))
    
        # Actualisation tableau de coordonnées avec coordos x,y,z si analyse effectuée
        if coordo_xyz:
            self.ids.grid.clear_widgets()
            self.ids.grid.cols = 4
            self.ids.grid.rows = 1 + nb_marqueurs
            self.ids.grid.size_hint = (.27, .24)
            self.ids.grid.add_widget(Label(text='Marqueurs', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text='Positions', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text='Marqueurs', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text='Coordonnées (x,y,z)', color=(0,0,0,1)))
            d_im = dict_coordo_xyz_labels[f'image{image_nb}']

            for key, l in zip(d_im.keys(), labels):
                self.ids.grid.add_widget(Label(text=f'{l}', color=(0,0,0,1)))
                if l in dict_coordo_labels_manual[f'image{image_nb}']:
                    p = dict_coordo_labels_manual[f'image{image_nb}'][l]
                else:
                    p = [np.nan, np.nan]
                self.ids.grid.add_widget(Label(text=f'({p[0]:.0f}, {p[1]:.0f})', color=(0,0,0,1)))
                self.ids.grid.add_widget(Label(text=f'{key}', color=(0,0,0,1)))
                c = d_im[key]
                self.ids.grid.add_widget(Label(text=f'({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f})', color=(0,0,0,1)))
            self.ids.origine.text = ''
            self.ids.width.text = ''
            self.ids.height.text = ''
        
        # Efface les marques de labellisation manuelle (ronds bleus) si désactivé
        if self.ids.labelize_manual.state == 'normal':
            self.canvas.remove_group(u"label")
        
    # Fonction pour détecter les marqueurs de toutes les images du répertoire de 
    def detect_marqueurs(self):
        timer_debut_detection = time.process_time_ns()
        if not os.path.exists(os.path.join(path,"annotated_frames","annotated_frame_0000.jpg")):
            warnings.warn("La première frame doit être annotée manuellement et enregistrée")
            return

        global detection_eff
        if len(path) > 1:
            # Détecte les marqueurs
            all_key_points = marker_detection_with_particles.annotate_frames_with_particles(path)

        global dict_coordo
        for i,frame_key_points in enumerate(all_key_points):
            #marker_array[0][i] = [[point.pt[0], point.pt[1]] for point in points]
            dict_coordo.update({f'image{i+1}' : [[float(point.pt[0]), float(point.pt[1])] for point in frame_key_points]})
            
        detection_eff = True

        # Va chercher les positions corrigées enregistrées si mode Ouvrir
        if self.ids.check_open.state == 'down':
            #global analyse_eff
            #analyse_eff = 'Metriques' in os.listdir(path+'') #Analyse effectuée (et utilisable) si métriques enregistrées

            global labelize_extent
            labelize_extent = True

            if 'coordonnees_xyz.csv' in os.listdir(path+'/Positions/'):
            # Recrée le dictionnaire de coordonnées x,y,z
                global dict_coordo_xyz_labels
                dict_coordo_xyz_labels = {}
                with open(path+'/Positions/coordonnees_xyz.csv', 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';')
                    j = 0
                    for row in reader: #skip headline
                        if j == 0:
                            entete = row[1::3]
                            labels_xyz = [e[:-2] for e in entete]
                            print(labels_xyz)
                        elif j > 0:
                            key = f'image{row[0]}'
                            dict_coordo_xyz_labels.update({key: {}})
                            row = [float(i) for i in row[1:]]
                            i = 0
                            for l in labels_xyz:
                                dict_coordo_xyz_labels[key].update({l : [row[i], row[i+1], row[i+2]]})
                                i += 3
                        j += 1

                global coordo_xyz
                coordo_xyz = True

            else:
                self.coordo_xyz_marqueurs()
            

        timer_fin_detection = time.process_time_ns()
        print(timer_debut_detection, timer_fin_detection)
        print(f'Temps détection des marqueurs :{timer_fin_detection - timer_debut_detection} ns')

        self.extend_labelisation()

    # Fonction pour afficher les marqueurs sur l'image actuelle
    def show_marqueurs(self):
        # Affichage des marqueurs si bouton activé
        if detection_eff == True:
            for coordinates in dict_coordo[f'image{image_nb}']:
                x = (coordinates[0]/im_dim[1])*(self.ids.image_show.width/self.width) + 0.025 # calcul des coordonnées sur l'écran à partir de celles sur l'image
                y = 0.85 - (coordinates[1]/im_dim[0])*0.78
                with self.canvas:
                    Color(0,1,0,1)
                    Line(circle=(self.width*x, self.height*y,6,0,360), width=1.1, group=u"circle") #(center_x, center_y, radius, angle_start, angle_end, segments)
        # Efface les marqueurs si bouton désactivé
        if self.ids.button_showmarks.state == 'normal':
            self.canvas.remove_group(u"circle")

    # Fonction pour vérifier le nombre de marqueurs détectés pour toutes les images du répertoire
    # (avec dictionnaire de coordonnées créé)
    # Retourne la liste des images n'ayant pas 5 marqueurs détectés et les liste dans une nouvelle fenêtre
    def verif_nb(self):
        im_prob_nb = [] #liste des images avec ±5 marqueurs
        for im, coordo in dict_coordo.items():
            if len(coordo) != nb_marqueurs:
                im_prob_nb.append(int(im[5:])) #ajoute l'image et le nombre de marqueurs détectés à la liste
        for im, dict in dict_coordo_labels_manual.items():
                coordo = list(dict.values())
                if [np.nan, np.nan] in coordo and int(im[5:])not in im_prob_nb:
                    im_prob_nb.append(int(im[5:]))
        
        im_prob_nb = sorted(im_prob_nb)

        if len(im_prob_nb) > 0:
            txt = f"Numéros des images n'ayant pas {nb_marqueurs} marqueurs :\n"
            txt_multiline = ''
            i = len(txt)
            count = 0
            # formattage du texte à afficher (lignes multiples)
            for im in im_prob_nb:
                txt += f'{im}, '
            for n in range(len(txt)):
                if txt[n]==',':
                    count+=1
                    if count == 30:
                        txt_multiline = txt[:n+1] + '\n' + txt[n+2:]
                    if count > 30 and count%30 == 0:
                        txt_multiline = txt_multiline[:n+1] + '\n' + txt_multiline[n+2:]
            if len(txt_multiline) > 1:
                self.ids.im_prob_nb.text = txt_multiline[:-2]
            else:
                self.ids.im_prob_nb.text = txt[:-2]
        elif len(im_prob_nb) == 0:
            self.ids.im_prob_nb.text = f'{nb_marqueurs} marqueurs détectés sur toutes les images !'
        return im_prob_nb
   
    # Fonction pour convertir la position touchée en coordonnées de marqueur, puis choisir l'action à exécuter (delete or add)
    def pos_marqueur(self, touch_pos):
        if not path:
            warnings.warn("Aucune image détectée, merci de selectionner un dossier pour commencer")
            return
        if self.ids.labelize_manual.state == 'normal':
            m_pos = [0,0]
            # im_dim = (600, 500, 3) = (height, width, channels)
            if 0.025*self.width <= touch_pos[0] <= (self.ids.image_show.width+0.025*self.width) and 0.07*self.height <= touch_pos[1] <= 0.85*self.height:
                m_pos[0] = (touch_pos[0]/self.width - 0.025)/(self.ids.image_show.width/self.width)*im_dim[1]
                m_pos[1] = -(touch_pos[1]/self.height - 0.85)/0.78*im_dim[0]
            # Détermine s'il y a un marqueur à effacer ou si on en ajoute un manquant
                add_m = False
                for c in dict_coordo[f'image{image_nb}']:
                    if abs(m_pos[0]-c[0]) < 10 and abs(m_pos[1]-c[1]) < 10:
                        dict_coordo[f'image{image_nb}'].remove(c)
                        if f'image{image_nb}' in dict_coordo_labels_manual:
                            dict_coordo_labels_manual[f'image{image_nb}'] = {m: c for m, c in dict_coordo_labels_manual[f'image{image_nb}'].items() if c in dict_coordo[f'image{image_nb}']}
                        if labelize_extent == True:
                            self.extend_labelisation()
                        if self.ids.button_verif_continuity.state == 'down':
                            self.verif_continuity()
                        self.show_image()
                        add_m = False
                        break
                    else:
                        add_m = True
                if add_m or len(dict_coordo[f'image{image_nb}']) == 0:
                        self.add_marqueur(m_pos)
            else:
                pass

    # Ajout manuel d'un marqueur sur commande par un clic sur l'image
    def add_marqueur(self, m_pos):
        x = (m_pos[0]/im_dim[1]*(702/1960)+0.02)
        y = (0.85 - m_pos[1]/im_dim[0]*0.78)
        with self.canvas:
            Color(0,0,1,1)
            d = 5
            Ellipse(pos=(x - d/2, y - d/2), size=(d, d), group=u"new_mark")
        dict_coordo[f'image{image_nb}'].append([m_pos[0], m_pos[1]])

        if labelize_extent == True:
            self.extend_labelisation()

        global detection_eff
        detection_eff = True
        self.show_image()
        
    # Supprime les marqueurs qui n'ont pas été identifiés dans la prolongation de la labellisation manuelle
    def delete_by_continuity(self):
        for im in dict_coordo.keys():
            for c in dict_coordo[im]:
                if c not in dict_coordo_labels_manual[im].values():
                    dict_coordo[im].remove(c)
        self.show_image()
    
    def validate_labels(self):
        for im in dict_coordo.keys():
            if im not in dict_coordo_labels_manual:
                dict_coordo_labels_manual[im] = {}
        
            for label in labels:  # Vérifie tous les labels attendus
                if label not in dict_coordo_labels_manual[im]:
                    dict_coordo_labels_manual[im][label] = [np.nan, np.nan]

    # Ajoute des marqueurs manquants selon les splines d'interpolation
    def add_by_continuity(self):
        im_prob_nb = self.verif_nb()
        splines_smooth = self.interpolate_spline()
        self.validate_labels()
        for im in im_prob_nb:
            for l in labels:
                c = dict_coordo_labels_manual[f'image{im}'][l]
                if len(splines_smooth[l][0]) > 1:
                    c_interpolate = [float(splines_smooth[l][0][im-1]), float(splines_smooth[l][1][im-1])]
                    if c == [np.nan, np.nan]:
                        dict_coordo[f'image{im}'].append(c_interpolate)
                        dict_coordo_labels_manual[f'image{im}'][l] = c_interpolate
                    # Modification des marqueurs loin de leur courbe d'interpolation
                    if (abs(c[0] - c_interpolate[0]) > 5 or abs(c[1] - c_interpolate[1]) > 5) and c in dict_coordo[f'image{im}']:
                        dict_coordo[f'image{im}'].remove(c)
                        dict_coordo[f'image{im}'].append(c_interpolate)
                        dict_coordo_labels_manual[f'image{im}'][l] = c_interpolate
        self.show_image()
    
    # Fonction pour détecter les discontinuités (changement important de pente) ...de moins en moins utile avec l'interpolation
    def verif_continuity(self):
        im_prob_continuity = {}
        for im in range(2, images_total):
            coordo_prec = dict_coordo_labels_manual[f'image{im-1}']
            coordo_act = dict_coordo_labels_manual[f'image{im}']
            coordo_next = dict_coordo_labels_manual[f'image{im+1}']
            for label in labels:
                if abs((coordo_next[label][0]-coordo_act[label][0])-(coordo_act[label][0]-coordo_prec[label][0])) > 10:
                    if im not in im_prob_continuity:
                        im_prob_continuity.update({im: [label]})
                    else:
                        im_prob_continuity[im].append(label)
                if abs((coordo_next[label][1]-coordo_act[label][1])-(coordo_act[label][1]-coordo_prec[label][1])) > 10:
                    if im not in im_prob_continuity:
                        im_prob_continuity.update({im: [label]})
                    elif label not in im_prob_continuity[im]:
                        im_prob_continuity[im].append(label)

        return im_prob_continuity
    
    # Fonction pour définir une spline d'inteprolation pour chaque position x, y des marqueurs
    def interpolate_spline(self):
        x_axis = np.arange(images_total)
        x, y = {}, {}
        splines_smooth, spl = {}, {}
        for l in labels:
            y.update({l : [[], []]})
            x.update({l: []})
            spl.update({l: [[], []]})
            splines_smooth.update({l: [[], []]})
        for im, coordos in dict_coordo_labels_manual.items():
            for l, c in coordos.items():
                if not math.isnan(c[0]):
                    y[l][0].append(c[0])
                    y[l][1].append(c[1])
                    x[l].append(int(im[5:]))

        for l in labels:
            m = len(x[l])
            if m > images_total/7:
                y[l][0] = gaussian_filter1d(y[l][0], 3) #filtre les données avant interpolation
                y[l][1] = gaussian_filter1d(y[l][1], 3)

                spl[l][0] = splrep(x[l], y[l][0], k=3)
                spl[l][1] = splrep(x[l], y[l][1], k=3)
                splines_smooth[l][0] = splev(x_axis, spl[l][0], ext=3)
                splines_smooth[l][1] = splev(x_axis, spl[l][1], ext=3)
            else:
                splines_smooth[l][0] = np.empty((images_total, ))
                splines_smooth[l][0][:] = np.nan
                splines_smooth[l][1] = np.empty((images_total, ))
                splines_smooth[l][1][:] = np.nan

        return splines_smooth

    # Graphiques des positions des marqueurs selon l'image, avec splines d'interpolation
    def graph_continuity(self):
        xaxis = range(1, images_total+1)
        splines_smooth = self.interpolate_spline()
        fig, (ax1, ax2) = plt.subplots(2,1)
        colors = ['tab:orange', 'tab:red', 'tab:green', 'k', 'tab:blue', 'tab:purple', 'y', 'c', 'tab:gray', 'tab:pink']
        for m, color in zip(labels, colors[0:nb_marqueurs]):
            try:
                plot_x = [c[m][0] for c in dict_coordo_labels_manual.values()]
                plot_y = [c[m][1] for c in dict_coordo_labels_manual.values()]
            except KeyError:
                continue
            ax1.scatter(xaxis, plot_x, s=2, label=m, c=color)
            ax1.plot(xaxis, splines_smooth[m][0], c=color)
            ax2.scatter(xaxis, plot_y, s=2, label=m, c=color)
            ax2.plot(xaxis, splines_smooth[m][1], c=color)
        ax1.legend(loc='center right', bbox_to_anchor=(1.13, -0.2), fontsize=9, frameon=False)
        ax1.set_title("Coordonnées des marqueurs selon l'image", fontsize=10)
        ax1.set_ylabel("Coordonnée en x", fontsize=9)
        ax2.set_ylabel("Coordonnée en y", fontsize=9)
        ax2.set_xlabel("Numéro de l'image", fontsize=9)

        self.ids.graph.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        plt.close()

        if self.ids.button_graph_continuity.state == 'normal':
            self.ids.graph.clear_widgets()

    # Groupes de fonctions pour labellisation manuelle sur une image, puis étendue sur les autres par proximité
    # Détecte clic et associe marqueur à labelliser
    def labelize_manual(self, touch_pos):
        global m_to_label
        m_to_label = [np.nan, np.nan]

        if self.ids.labelize_manual.state == 'down':
            m_pos = [0,0]
            # im_dim = (600, 500, 3) = (height, width, channels)
            # détecte si le clic est dans le cadre de l'image
            if 0.025*self.width <= touch_pos[0] <= (self.ids.image_show.width+0.025*self.width) and 0.07*self.height <= touch_pos[1] <= 0.85*self.height:
                m_pos[0] = (touch_pos[0]/self.width - 0.025)/(self.ids.image_show.width/self.width)*im_dim[1]
                m_pos[1] = -(touch_pos[1]/self.height - 0.85)/0.78*im_dim[0]
            # Trouve le marqueur le plus près pour lui associer le label
                for c in dict_coordo[f'image{image_nb}']:
                    if abs(m_pos[0]-c[0]) < 20 and abs(m_pos[1]-c[1]) < 20:
                        # ajout d'un cercle bleu pâle pour montrer qu'un marqueur est sélectionné
                        with self.canvas:
                            Color(171/255.0, 222/255.0, 231/255.0, .8)
                            d = 7
                            Ellipse(pos=(touch_pos[0] - d/2, touch_pos[1] - d/2), size=(d, d), group=u"label")
                        
                        m_to_label = c
                        break   
        else:
            pass
    
    def modify_label(self, label,pos):
        for i,widget in enumerate(self.ids.grid.children):
            if ''.join(list(widget.text)) == label:
                self.ids.grid.children[i-1].text = f'({pos[0]:.0f}, {pos[1]:.0f})'
                return 
        
        raise ValueError(f"No label with name {label} found")
    
    def add_label(self,label,pos):
        self.ids.grid.add_widget(Label(text=f'{label}', color=(0,0,0,1)))
        self.ids.grid.add_widget(Label(text=f'({pos[0]:.0f}, {pos[1]:.0f})', color=(0,0,0,1)))


    # Entre le label du marqueur sélectionne dans le tableau et dans le dictionnaire, extend labelisation si tous les marqueurs labellisés
    def label_in(self, button):
        id = button.custom_value
        label = id[5:]

        self.canvas.remove_group(u"label")

        if m_to_label != [np.nan, np.nan]:
            global dict_coordo_labels_manual
            update = False
            if f'image{image_nb}' in dict_coordo_labels_manual.keys():
                if label not in dict_coordo_labels_manual[f'image{image_nb}']:
                    for key,value in dict_coordo_labels_manual[f'image{image_nb}'].items():
                        if m_to_label == value:
                            warnings.warn(f"Position touchée ({m_to_label}) déjà utilisée pour le label {key}")
                    dict_coordo_labels_manual[f'image{image_nb}'].update({label : m_to_label})
                else:
                    dict_coordo_labels_manual[f'image{image_nb}'][label] = m_to_label
                    update = True
            else:
                dict_coordo_labels_manual.update({f'image{image_nb}': {label : m_to_label}})

            if not labelize_extent:
                if update:
                    self.modify_label(label,m_to_label)
                else:
                    self.add_label(label,m_to_label)
            if labelize_extent:
                self.show_image()
            
            if len(list(dict_coordo_labels_manual[f'image{image_nb}'].keys())) == nb_marqueurs:
                global labels
                labels = list(dict_coordo_labels_manual[f'image{image_nb}'].keys())
                
                self.extend_labelisation()

                # reinit buttons for further manual labelization
                self.ids.marq_C7.state = 'normal'
                self.ids.marq_Tsup.state = 'normal'
                self.ids.marq_Tap.state = 'normal'
                self.ids.marq_Tinf.state = 'normal'
                self.ids.marq_Lap.state = 'normal'
                self.ids.marq_Linf.state = 'normal'
                self.ids.marq_ScG.state = 'normal'
                self.ids.marq_ScD.state = 'normal'
                self.ids.marq_IG.state = 'normal'
                self.ids.marq_ID.state = 'normal'
   
        if np.isnan(nb_marqueurs):
            print('color')
            self.ids.nb_marqueurs.color = (1,0,0,1)
    
    # Étend la labellisation de la 1re image aux suivantes (réexécutée en cours de correction)
    def extend_labelisation(self):
        global labelize_extent
        labelize_extent = True
        for im in list(sorted(dict_coordo.keys(), key=lambda item : int(item[5:])))[1:]:
            num = int(im[5:])
            if im not in dict_coordo_labels_manual:
                dict_coordo_labels_manual.update({im:{}})
            # définit les références à partir des positions correspondantes sur les images précédentes
            for label in dict_coordo_labels_manual['image1'].keys():
                ref_prec = [0,0]
                if label in dict_coordo_labels_manual[im] and dict_coordo_labels_manual[im][label] != [np.nan, np.nan]:
                    continue
                else:
                    i = 1
                    while i < num:
                        if label in dict_coordo_labels_manual[f'image{num-i}'] and dict_coordo_labels_manual[f'image{num-i}'][label] != [np.nan, np.nan]:
                            ref = dict_coordo_labels_manual[f'image{num-i}'][label]
                            if num > 2:
                                j = i+1
                                while j <= (num-i):
                                    if label in dict_coordo_labels_manual[f'image{num-j}'] and dict_coordo_labels_manual[f'image{num-j}'][label] != [np.nan, np.nan]:
                                        ref_prec = dict_coordo_labels_manual[f'image{num-j}'][label]
                                        break
                                    else:
                                        j += 1
                            break
                        else:
                            i += 1
                    # cherche dans les marqueurs du dictionnaire de l'image actuelle s'il y en a un qui peut être labellisé (proche de ref ou ref_prec)
                    for coordos in dict_coordo[im]:
                        if coordos not in dict_coordo_labels_manual[im].values():
                            if (abs(coordos[0] - ref[0]) < 12 and abs(coordos[1] - ref[1]) < 10) or (abs(coordos[0] - ref_prec[0]) < 12 and abs(coordos[1] - ref_prec[1]) < 10):
                                dict_coordo_labels_manual[im][label] = coordos                            
                                break
                            elif ref_prec != [0,0] and -7 < (coordos[0] - (i*(ref_prec[0]-ref[0])/(j-i)+ref[0])) < 9 and -7 < (coordos[1] - (i*(ref_prec[1]-ref[1])/(j-i)+ref[1])) < 9:
                                dict_coordo_labels_manual[im][label] = coordos
                                break
                        else:
                            dict_coordo_labels_manual[im][label] = [np.nan, np.nan]
        
        self.ids.button_verif_nb.disabled = False
        self.ids.button_verif_continuity.disabled = False
        self.ids.button_graph_continuity.disabled = False
        self.ids.button_delete.disabled = False
        self.ids.button_interpolate.disabled = False         
                        
    # Fonction pour extraire les coordonnées (x,y,z) des marqueurs des fichiers _xyz_.raw
    def coordo_xyz_marqueurs(self):
        global dict_coordo_xyz_labels
        dict_coordo_xyz_labels = {}
        global save_xyz
        save_xyz = path+'/XYZ_converted/'
        os.makedirs(save_xyz, exist_ok=True)
        # Lis les xyz.raw et crée les fichiers contenant les x,y,z des marqueurs

        RRF.write_xyz_coordinates(path, dict_coordo_labels_manual, w1, w2, h1, h2)
        # Récupère les données des fichiers csv des coordonnées x,y,z des marqueurs
        for filename in os.listdir(save_xyz):
            index_XYZ = filename.find('_XYZ') + 5
            key = f'image{int(filename[index_XYZ:-4])+1}'
            with open(os.path.join(save_xyz, filename), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                dict_coordo_xyz_labels.update({key : {}})
                for row in reader:
                    l = row[0]
                    row = [float(i) for i in row[1:]]
                    dict_coordo_xyz_labels[key].update({l:[row[1], row[0], row[2]]})

        global coordo_xyz
        coordo_xyz = True

    # Sauvegarder les informations souhaitées selon ce qui est coché
    def to_save(self):
        timer_debut_save = time.process_time_ns()
      
        if not 'Positions' in os.listdir(path):
            os.mkdir(path+'/Positions', )
        if not 'annotated_frames' in os.listdir(path):
            os.mkdir(path+'/annotated_frames', )
        self.save_positions()

        timer_fin_save = time.process_time_ns()
        print(timer_debut_save, timer_fin_save)
        print(f'Temps sauvegarde :{timer_fin_save - timer_debut_save} ns')
    
    # Crée un csv et y écrit les coordonnées x,y,z des 5 marqueurs selon le numéro de l'image
    def save_positions(self):
        save_pos = path+'/Positions'
        print('writing to ' + save_pos+'/positions_corrigees.json')
        with open(save_pos+'/positions_corrigees.json', 'w') as positions:
            json.dump(dict_coordo_labels_manual, positions)

        #ne sauvegarde que les imqges posseddant le bon nombre de marqueurs
        print("Ecriture des images annotées")
        nb_saved_annotated = 0
        annotated_frames_path = path + '/annotated_frames'
        for i, (filename,coordo_list) in enumerate(zip(os.listdir(save_path_im),dict_coordo.values())):
            if len(coordo_list) == nb_marqueurs:
                nb_saved_annotated += 1 
                #Save annotated image

                preprocessed_frame = cv2.imread(os.path.join(save_path_im, filename), cv2.IMREAD_GRAYSCALE)
                key_points = [cv2.KeyPoint(int(np.round(midpoint[0])), int(np.round(midpoint[1])), 15) for midpoint in coordo_list]
                annotated_file = f"annotated_frame_{i:04d}.jpg"
                frame_with_key_points = cv2.drawKeypoints(preprocessed_frame, key_points, None, color=(0, 255, 0))
                cv2.imwrite(os.path.join(annotated_frames_path, annotated_file), frame_with_key_points)
        
        if os.path.exists(os.path.join(path,"annotated_frames","annotated_frame_0000.jpg")):
            self.ids.button_particle_filter.disabled = False

        
class Interface(App):
    def build(self):
        return MyApp()

if __name__=='__main__':
    Interface().run()
