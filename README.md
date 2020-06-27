## Modèle
Mise en Place d'un modèle prédictif de la valeur foncière lors d'une vente d'un appartement ou d'une maison

## Fichiers 
Les fichiers trop volumineux sont dans le .gtignore

## Librairies
Installer les librairies via `pip install -r requirements.txt`

## Architecture du Projet :

- Projet (Preprocessing,Modeles,Config,Results,Data)
- requirements.txt
- Main.py
- Presentation.ipynb

## Choix des Paramètres
Les colonnes des données que j'ai considéré utiles à la prédiction :

- Valeur Foncière : Target donc obligatoire
- Code département : Prix des maisons et des appartements dépendent beaucoup du département
- Code commune : Peut-être trop précis
- Nombre de lots dans une maison 
- Les surfaces et type de lots
- Type local
- Surface réelle
- Nombre de pièces principales
- Surface terrain

## Échantillon aléatoire

Étant donné la taille de la donnée, nous allons entraîner les modèles sur un échantillon aléatoire et donc nous supposons que cet échantillon peut représenter toutes les données au complet