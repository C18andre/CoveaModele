## Modèle

Mise en Place d'un modèle prédictif de la valeur foncière lors d'une vente d'un appartement ou d'une maison

Pour l'utilisation du projet, se référer au UserGuide.md

## Architecture du Projet

- Projet (Preprocessing,Modeles,Config,Results,Data)
- requirements.txt
- Main.py
- Presentation.ipynb
- UserGuide.md : Guide d'utilisation du modèle
- Les fichiers trop volumineux sont dans le .gitignore donc pas présent sur github

## Documentation

En premier lieu, je me suis documenté sur ce qui impactait la valeur d'un logement

Les critères principaux sont :
 - La situation du bien
 - Son état
 - Terrain extéreur
 - Étage (Pour les appartements)
 - Différents lots

Références :

 - [1] : https://immobilier.lefigaro.fr/annonces/edito/acheter/les-prix-de-l-immobilier/immobilier-comment-estimer-plus-facilement-votre-bien-grace-au-big-data

 - [2] : https://www.economie.gouv.fr/cedef/estimer-prix-immobilier

## Filtrage des données et choix des paramètres

 - Suppression des colonnes vides
 - Suppression des doublons (lignes identiques)
 - On garde uniquement les ventes sur des Maisons et Appartements
 - Afin de s'assurer de la qualité des données, je choisis de supprimer les ventes de moins de 10 000€
 - Choix d'une ville, car les prix sont très dépendant de la situation géographique d'un bien
 - Je choisis également de supprimer les ventes de plus de 1 500 000€ (Au dessus de ce prix, d'autres paramètres rentrent en compte)
 Ensuite nous avons donc les données pour chaque ville (Les plus grandes)

Après avoir choisit une ville,
Les colonnes des données que j'ai considéré utile à la prédiction :
( Celles qui se rapprochent le plus des critères principaux)

- Valeur Foncière : Target donc obligatoire
- Nombre de lots dans une maison 
- Les surfaces Carrez des différents lots (Plus de 1m80 de plafond)
- Type local
- Type de voie (Encoder avec LabelEncoder())
- Code Commune (Encoder avec LabelEncoder())
- Surface réelle
- Nombre de pièces principales
- Surface terrain

## Normalisation des données

Afin que le modèle puisse apprendre au vu de la disparité des valeurs, je choisis d'appliquer la fonction ln(1+X) sur la valeur fonciere,
cette fonction va agir comme un filtre passe bas et diminuer fortement les disparités au sein de l'échantillon de donnée. On va ainsi obtenir une répartition quasi-normale.

## Différents modèles utilisés

 Je propose de comparer 3 modèles fréquemment utilisé dans les problèmes de régression :

 - Neural network (J'ai choisi ici un réseau de neurones classique composé de 3 couches)
 - XGBoost (J'ai choisi d'utiliser également le xgb, que j'avais déjà utilisé pour un problème de régression)
 - RandomForestRegressor (Le random forest regressor que j'ai également déjà utilisé)

 Documentation et projet similaire :

 [1] : https://www.kaggle.com/erick5/predicting-house-prices-with-machine-learning
 



