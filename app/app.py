import streamlit as st
import os
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable

# =========================================
# Fonctions personnalisées pour le modèle
# =========================================
@register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    intersection = np.sum(y_true * y_pred, axis=[1, 2, 3])
    union = np.sum(y_true, axis=[1, 2, 3]) + np.sum(y_pred, axis=[1, 2, 3])
    return (2. * intersection + smooth) / (union + smooth)

@register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

@register_keras_serializable()
def bce_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

@register_keras_serializable()
def iou_metric(y_true, y_pred):
    y_true = np.round(y_true).astype(bool)
    y_pred = np.round(y_pred).astype(bool)
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))
    return (intersection + 1e-10) / (union + 1e-10)

# Charger le modèle
MODEL_PATH = "./models/U-Net Miniaug.keras"
try:
    model = load_model(MODEL_PATH, custom_objects={
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss,
        "bce_loss": bce_loss,
        "iou_metric": iou_metric
    })
    model_loaded = True
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    model_loaded = False

# Configuration du modèle
INPUT_SIZE = (256, 256)

PALETTE = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
]

CLASS_LABELS = [
    "Flat",
    "Human",
    "Vehicle",
    "Construction",
    "Object",
    "Nature",
    "Sky",
    "Void",
]

def apply_palette(mask, palette):
    """
    Applique une palette de couleurs à un masque d'indices de classes.
    """
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        color_mask[mask == class_id] = color
    return color_mask

# =========================================
# Page d'accueil
# =========================================
def display_home():
    st.markdown(
        """
        <style>
        .header {
            font-family: 'Arial', sans-serif;
            text-align: center;
            padding: 50px 20px;
            background-color: #f7f7f7;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .header h1 {
            color: #333;
            font-size: 3rem;
            margin-bottom: 10px;
        }
        .header h2 {
            color: #666;
            font-size: 1.5rem;
            margin-bottom: 20px;
        }
        .disclaimer {
            font-family: 'Arial', sans-serif;
            background-color: #ffefef;
            border-left: 5px solid #ff6f6f;
            padding: 15px;
            margin: 20px auto;
            border-radius: 5px;
            color: #333;
            line-height: 1.5;
        }
        .disclaimer strong {
            font-weight: bold;
        }
        </style>
        <div class="header">
            <h1>Bienvenue sur SOPHIA</h1>
            <h2>application de Segmentation et Observation Prédictive d'Images pour l'Automobile</h2>
        </div>
        <div class="disclaimer">
            <strong>Disclaimer :</strong> Les images uploadées sur cette application ne sont en aucun cas sauvegardées. Veuillez les utiliser en toute confiance.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================
# Upload et prédiction
# =========================================
def display_upload_and_predict():
    st.title("Uploader une Image")
    uploaded_file = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image Uploadée", use_container_width=True)

        if model_loaded:
            st.write("**Analyse de l'image**")
            analyze_button = st.button("Analyser l'image")

            if analyze_button:
                # Prétraitement de l'image
                img_resized = ImageOps.fit(image, INPUT_SIZE, Image.Resampling.LANCZOS)
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prédiction du masque
                prediction = model.predict(img_array)[0]
                predicted_mask = np.argmax(prediction, axis=-1)

                # Application de la palette
                predicted_mask_colored = apply_palette(predicted_mask, PALETTE)

                st.image(predicted_mask_colored, caption="Masque Prédit", use_container_width=True)
        else:
            st.error("Le modèle n'a pas été chargé correctement.")

# =========================================
# Observation des données d'entraînement
# =========================================
def display_data_analysis():
    st.title("Observation des données d'entraînement")
    st.write("Sélectionnez une ville et une image pour observer les masques correspondants.")

    cities = sorted(os.listdir("api_image"))
    selected_city = st.selectbox("Choisissez une ville", cities)

    # Liste des images disponibles pour la ville sélectionnée
    city_images_path = os.path.join("api_image", selected_city)
    images = sorted([img for img in os.listdir(city_images_path) if img.endswith(".png")])
    selected_image = st.selectbox("Choisissez une image", images)

    if selected_image:
        # Chemins des fichiers image et masque
        image_path = os.path.join(city_images_path, selected_image)
        mask_path = os.path.join("api_mask", selected_city, selected_image.replace("_leftImg8bit.png", "_gtFine_labelIds.png"))

        if os.path.exists(image_path) and os.path.exists(mask_path):
            # Chargement de l'image et du masque d'origine
            image = Image.open(image_path)
            original_mask = Image.open(mask_path).convert("L")

            st.write("Image originale et masque correspondant :")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Image Originale", use_container_width=True)
            with col2:
                st.image(original_mask, caption="Masque Original (Niveaux de gris)", use_container_width=True)

            # Analyse avec le modèle
            if model_loaded:
                st.write("**Analyse du masque avec le modèle**")
                analyze_button = st.button("Analyser")

                if analyze_button:
                    # Prétraitement de l'image
                    img_resized = ImageOps.fit(image, INPUT_SIZE, Image.Resampling.LANCZOS)
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Prédiction du masque
                    prediction = model.predict(img_array)[0]
                    predicted_mask = np.argmax(prediction, axis=-1)

                    # Redimensionnement du masque traité à la taille originale
                    predicted_mask_resized = Image.fromarray(predicted_mask.astype(np.uint8))
                    predicted_mask_resized = predicted_mask_resized.resize(image.size, Image.Resampling.NEAREST)

                    # Application de la palette
                    predicted_mask_colored = apply_palette(np.array(predicted_mask_resized), PALETTE)

                    # Affichage du masque traité et de sa légende
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(predicted_mask_colored, caption="Masque Traité", use_container_width=True)

                    with col2:
                        st.write("**Légende des classes avec couleurs**")
                        for class_id, class_name in enumerate(CLASS_LABELS):
                            st.markdown(
                                f'<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                                f'<div style="width: 20px; height: 20px; background-color: rgb{PALETTE[class_id]}; '
                                f'margin-right: 10px; border: 1px solid #000;"></div>'
                                f'<span>{class_name}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
        else:
            st.error("Fichiers image ou masque manquants pour cette sélection.")

# =========================================
# Barre de navigation
# =========================================
st.sidebar.title("Navigation")
pages = {
    "Accueil": display_home,
    "Uploader et Prédire": display_upload_and_predict,
    "Observation des Données": display_data_analysis
}
selection = st.sidebar.radio("Aller à", list(pages.keys()))
pages[selection]()
