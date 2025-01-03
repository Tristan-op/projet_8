
# **Projet SOPHIA**

## **Description**
Le projet SOPHIA (Segmentation d'Objets et Prédiction pour l'Habilitation d'Intelligence Artificielle) est une solution complète de segmentation d'images pour les applications de conduite autonome.  
Ce projet contient deux principaux composants :

1. Une API FastAPI pour l'exposition du modèle entraîné.
2. Une application Streamlit pour la visualisation des résultats.

---

## **Structure du Projet**

### **1. Dossier `api`**
Ce dossier contient les fichiers nécessaires pour l'API **FastAPI**.  
L'API expose les fonctionnalités suivantes :
- **Liste des images disponibles** : Obtenez les noms des images et masques stockés localement.
- **Observation des données** : Visualisez une image et son masque annoté.
- **Prédiction du masque** : Envoyez une image en entrée, et recevez un masque prédictif généré par le modèle.

### **2. Dossier `app`**
Ce dossier contient l'application **Streamlit**, qui est une interface utilisateur conviviale consommant l'API.  
Les fonctionnalités de l'application Streamlit incluent :
- **Navigation intuitive** : Une barre latérale pour passer entre les pages.
- **Observation des données** : Affichez les images et masques disponibles.
- **Analyse interactive** : Téléchargez une image et visualisez les masques annotés et prédits.

---

## **Comment Utiliser ?**

### **Étape 1 : API FastAPI**
1. **Naviguez dans le dossier `api`** :
   ```bash
   cd api
   
2. Installez les dépendances :
    ```bash   
    pip install -r requirements.txt
3. Lancez l'API :
   ```bash 
   uvicorn main:app --reload

L'API sera accessible localement sur : http://127.0.0.1:8000


### **Étape 2 : Application Streamlit**
1. **Naviguez dans le dossier `app`** :   
   ```bash
   cd app

2. Installez les dépendances :
    ```bash   
    pip install -r requirements.txt
3. Lancez l'application :
   ```bash 
   streamlit run streamlit.py

L'application sera accessible sur : http://localhost:8501

### Déploiement

- **API FastAPI** : Déployée sur une plateforme Cloud comme **Azure**, **Heroku**, ou **PythonAnywhere**.
- **Application Streamlit** : Déployée sur **Streamlit Cloud**.

Les fichiers de configuration et scripts nécessaires pour le déploiement sont disponibles dans leurs dossiers respectifs :

- `api/` : Contient les fichiers pour l'API FastAPI.
- `app/` : Contient les fichiers pour l'application Streamlit.

