# Classification de Simpsons

Un adolescent a créé un modèle qui reconnaît des personnages des Simpsons :
Il a utilisé les modules Tensorflow et Keras en python pour créer son réseau de neurones.

## Stage IA à Magic Makers

[Magic Makers](https://www.magicmakers.fr/) propose des ateliers de programmation créative pour des jeunes de 7 à 15 ans. Depuis 2018, des ateliers pour adolescents autour de l'intelligence artifielle sont donnés durant les vacances. Lors du stage, les makers découvrent ce qu'est un réseaux de neurones et les notions s'y attachant (perceptron multi-couches, convolutions, overfit, etc) en créant des projets comme celui-ci !

## Auteur du projet

Ce projet a été réalisé par **Alexandre** lors du stage de Juillet dans les bureaux de Blablacar Paris 2e, animé par **Romain et Jade**.


### Dataset

* [Dataset de personnages des Simpsons](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset) - Les images de Simpsons pour l'entraînement


### Entraînement

Alexandre a commencé par sélectionner les photos de ses 3 personnages (Homer, Marge et Bart) pour son modèle. Il a ensuite utilisé un réseau de neurones par convolution pour son projet.

```
python3 simpsons-train.py
```
## Modules

* [Keras](https://keras.io/) - pour créer le modèle (avec TensorFlow)
* [Flask](http://flask.pocoo.org/) - pour créer une webapp
* [PIL](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html) - pour manipuler des images
* [Numpy](https://www.numpy.org/) - pour manipuler des tableaux
* [H5py](https://www.h5py.org/) - pour sauvegarder le modèle
* [Sklearn](https://scikit-learn.org/stable/) - pour mélanger et séparer les données

## Résultats

< à venir >

### Application et prédictions

Une fois son modèle entraîné, Alexandre a créé un programme pour prédire le personnage des Simpsons présent sur une photo !

```
python3 simpsons-predict.py
```

### Remerciements

* Merci à [Blablacar](https://www.blablacar.fr/) de nous avoir acceuilli dans vos locaux
* Merci à [Magic Makers](https://www.magicmakers.fr/)
* Merci à [Keras](https://keras.io/) pour faciliter la création de réseaux de neurones !
* Merci à [Kaggle](https://www.kaggle.com/) pour le dataset
