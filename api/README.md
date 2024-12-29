# **SOPHIA API**

## **Description**
L'API **SOPHIA** (Segmentation d'Objets et Prédiction pour l'Habilitation d'Intelligence Artificielle) permet de prédire des masques segmentés à partir d'images.  
Cette API est déployée pour faciliter l'utilisation du modèle de machine learning entraîné pour la segmentation d'images, notamment dans des applications telles que la conduite autonome.

---

## **Fonctionnalités**

### **1. Liste des images disponibles**
- Obtenez la liste des images et masques disponibles dans les dossiers `api_image` et `api_mask`.

### **2. Observation des données**
- Visualisez une image et son masque annoté.

### **3. Prédiction de masque**
- Envoyez une image, et recevez un masque prédictif généré par le modèle.

---

## **Installation**

### **Prérequis**
- **Python** : Version 3.12.
- **Bibliothèques nécessaires** : Installées via `requirements.txt`.

### **Étapes d'installation**
1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/votre-repo/sophia-api.git
   cd sophia-api
