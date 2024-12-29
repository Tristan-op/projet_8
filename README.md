SOPHIA: Segmentation d'Images pour les Voitures Autonomes
Bienvenue sur SOPHIA (Segmentation Optimisée pour la Prédiction et l'Hiérarchisation des Images en Autonomie), une application Streamlit permettant la segmentation d'images dans le contexte des voitures autonomes. Cette application utilise un modèle de segmentation d'images basé sur l'architecture U-Net, accompagné de fonctionnalités d'observation et d'analyse des données d'entraînement.

Fonctionnalités
1. Page d'accueil
Présente un disclaimer assurant aux utilisateurs que leurs images uploadées ne seront pas sauvegardées.
Explique l'objectif principal de l'application.
2. Upload et segmentation d'images
Les utilisateurs peuvent uploader une image de leur choix.
L'application effectue une segmentation de l'image grâce au modèle U-Net préchargé.
Le résultat de la segmentation est affiché en superposition avec une palette de couleurs correspondant aux classes (voir ci-dessous).
3. Observation des données d'entraînement
Permet de visualiser des images issues des ensembles de données d'entraînement.
Affiche l'image brute, le masque annoté (niveau de gris), et le masque traité (avec la segmentation appliquée par le modèle).
Propose une légende des classes sous forme de tableau, associant les couleurs aux labels.
4. Navigation simplifiée
Un menu latéral permet de naviguer facilement entre les différentes fonctionnalités.
Classes de segmentation
Voici les classes prises en charge par le modèle, avec leurs couleurs respectives utilisées dans les masques segmentés :

Classe	Couleur (RGB)
Flat	(0, 0, 0)
Human	(128, 0, 0)
Vehicle	(0, 128, 0)
Construction	(128, 128, 0)
Object	(0, 0, 128)
Nature	(128, 0, 128)
Sky	(0, 128, 128)
Void	(128, 128, 128)
Prérequis
Avant d'exécuter l'application, assurez-vous que votre environnement Python dispose des bibliothèques suivantes (ces dépendances sont listées dans le fichier requirements.txt) :

streamlit
tensorflow
numpy
pillow
matplotlib

Installez-les avec la commande :
pip install -r requirements.txt
