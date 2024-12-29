# API SOPHIA

L'API SOPHIA (Segmentation d'Objets et Prédiction pour l'Habilitation d'Intelligence Artificielle) est une API basée sur **FastAPI**. Elle permet de prédire des masques segmentés pour des images, ainsi que de lister et observer des données d'images et de masques.

## Fonctionnalités

### 1. Liste des images disponibles
- **Endpoint** : `/list-images`
- **Description** : Retourne la liste des images et masques disponibles organisés par villes dans les dossiers `api_image` et `api_mask`.

### 2. Observation des données
- **Endpoint** : `/observe`
- **Description** : Permet de visualiser une image et son masque annoté pour une ville et un nom d'image donnés.

### 3. Prédiction de masque
- **Endpoint** : `/predict`
- **Description** : Envoie une image, et retourne un masque prédictif généré par le modèle de segmentation.

---

## Utilisation

L'API déployée est disponible à l'adresse suivante : **[https://sophia.azure.net](https://sophia.azure.net)**

### 1. Accueil
- **Endpoint** : `/`
- Retourne une description des routes disponibles et leurs fonctionnalités.

### 2. Lister les images
- **Endpoint** : `/list-images`
- **Méthode** : `GET`
- Exemple de réponse :
  ```json
  {
    "city1": {
      "images": ["image1_leftImg8bit.png", "image2_leftImg8bit.png"],
      "masks": ["image1_gtFine_labelIds.png", "image2_gtFine_labelIds.png"]
    },
    "city2": {
      "images": ["image3_leftImg8bit.png"],
      "masks": ["image3_gtFine_labelIds.png"]
    }
  }
## Fonctionnalités

### 3. Observer une image et son masque annoté
- **Endpoint** : `/observe`
- **Méthode** : `GET`
- **Description** : Retourne une image et son masque annoté disponible dans les dossiers `api_image` et `api_mask`.

#### Paramètres :
- `city` : Ville dans laquelle se trouve l'image (exemple : `city1`).
- `image_name` : Nom de l'image (exemple : `image1_leftImg8bit.png`).

#### Exemple de requête :
    ```arduino
    https://sophia.azure.net/observe?city=city1&image_name=image1_leftImg8bit.png

Exemple de réponse :
L'API retourne les images directement en tant que contenu JPEG avec un type MIME approprié (image/jpeg).

### **4. Prédire un masque**
- **Endpoint** : `/predict`
- **Méthode** : `POST`
- **Description** : Traite une image envoyée pour générer un masque prédit au format JPEG.

#### **Paramètres** :
- Une image envoyée au format `multipart/form-data`.

#### **Exemple de requête avec `curl`** :
  ```bash
  curl -X POST "https://sophia.azure.net/predict" -F "file=@path_to_image.jpg"

Exemple de réponse :
   ```bash
  HTTP/1.1 200 OK
  Content-Type: image/jpeg

