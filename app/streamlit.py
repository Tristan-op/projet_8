import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
import numpy as np

# Configuration de l'application Streamlit
st.set_page_config(
    page_title="SOPHIA App",
    page_icon="🌟",
    layout="wide"
)

# Base URL de l'API (modifiez pour correspondre à votre déploiement cloud si nécessaire)
BASE_URL = "https://sophia-dkasd9g9dcfnaqb2.francecentral-01.azurewebsites.net/"


# Fonction utilitaire pour décoder une image Base64
def decode_base64_image(base64_string):
    image_bytes = BytesIO(base64.b64decode(base64_string))
    return Image.open(image_bytes)

# Pages de l'application
def page_home():
    st.title("Bienvenue sur SOPHIA 🌟")
    st.write("""
        SOPHIA est une application de segmentation d'images qui vous permet de visualiser, d'analyser et de prédire des masques pour des images.
        Utilisez le menu à gauche pour explorer les fonctionnalités :
        - Liste des images disponibles.
        - Upload et analyse de vos propres images.
    """)
    st.image("https://via.placeholder.com/800x400.png?text=SOPHIA+App+Overview", use_container_width=True)

def page_list_images():
    st.title("Liste des images disponibles")

    # Appel à l'API pour obtenir la liste des images
    response = requests.get(f"{BASE_URL}/list-images")

    if response.status_code == 200:
        data = response.json()
        cities = list(data.keys())

        # Sélection de la ville
        city = st.selectbox("Choisissez une ville :", options=cities)

        if city:
            images = data[city]["images"]
            st.write(f"Images disponibles pour la ville **{city}** :")

            # Sélection de l'image
            image_name = st.selectbox("Choisissez une image :", options=images)

            if image_name:
                # Aperçu de l'image et du masque
                st.write(f"Prévisualisation de l'image **{image_name}** :")
                preview_response = requests.get(
                    f"{BASE_URL}/view",
                    params={"city": city, "image_name": image_name, "accept": "json"}
                )

                if preview_response.status_code == 200:
                    result = preview_response.json()

                    # Décoder les images reçues
                    image = decode_base64_image(result["original_image"])
                    mask = decode_base64_image(result["annotated_mask"])

                    # Afficher les images empilées
                    st.image(image, caption="Image originale", use_container_width=True)
                    st.image(mask, caption="Masque annoté", use_container_width=True)

                    # Bouton pour analyser l'image
                    if st.button("Analyser cette image"):
                        with st.spinner("Analyse en cours..."):
                            analyze_response = requests.get(
                                f"{BASE_URL}/view",
                                params={"city": city, "image_name": image_name, "analyze": True, "accept": "json"}
                            )
                            if analyze_response.status_code == 200:
                                analyze_result = analyze_response.json()

                                # Décoder et redimensionner le masque prédit
                                predicted_mask = decode_base64_image(analyze_result["processed_mask"])
                                predicted_mask_resized = predicted_mask.resize(
                                    (predicted_mask.width // 4, predicted_mask.height // 4)
                                )

                                # Afficher le masque prédit
                                st.image(predicted_mask_resized, caption="Masque prédit (réduit à 25%)", use_container_width=True)

                                # Afficher la légende des classes
                                st.write("### Légende des classes :")
                                for label, color in zip(
                                    [
                                        "Flat",
                                        "Human",
                                        "Vehicle",
                                        "Construction",
                                        "Object",
                                        "Nature",
                                        "Sky",
                                        "Void",
                                    ],
                                    [
                                        (0, 0, 0),
                                        (128, 0, 0),
                                        (0, 128, 0),
                                        (128, 128, 0),
                                        (0, 0, 128),
                                        (128, 0, 128),
                                        (0, 128, 128),
                                        (128, 128, 128),
                                    ],
                                ):
                                    st.write(
                                        f"<span style='color:rgb({color[0]}, {color[1]}, {color[2]})'>🔸 {label}: RGB({color[0]}, {color[1]}, {color[2]})</span>",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.error("Erreur lors de la prédiction.")
                else:
                    st.error("Erreur lors de la récupération de l'image et du masque.")
    else:
        st.error("Impossible de récupérer la liste des images.")


def page_upload():
    st.title("Upload et analyse d'images")
    
    # Formulaire pour uploader une image
    uploaded_file = st.file_uploader("Choisissez une image :", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Aperçu de l'image uploadée
        image = Image.open(uploaded_file)
        st.image(image, caption="Image uploadée", use_container_width=True)
        
        # Bouton pour analyser l'image
        if st.button("Analyser l'image"):
            with st.spinner("Analyse en cours..."):
                # Envoi de l'image à l'API via POST
                files = {"file": uploaded_file.getvalue()}  # Obtenir le contenu brut
                response = requests.post(f"{BASE_URL}/predict", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Décodage des résultats
                    original_image = decode_base64_image(result["original_image"])
                    processed_mask = decode_base64_image(result["processed_mask"])
                    
                    # Affichage des résultats
                    st.image(original_image, caption="Image originale", use_container_width=True)
                    st.image(processed_mask, caption="Masque prédit", use_container_width=True)
                else:
                    st.error(f"Erreur lors de l'analyse de l'image : {response.status_code}")



# Menu de navigation
menu = {
    "Sommaire": page_home,
    "Liste des images": page_list_images,
    "Upload et analyse": page_upload
}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Menu", list(menu.keys()))
menu[choice]()
