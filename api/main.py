from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image, ImageOps
import tempfile
import base64

# Désactiver CUDA si non nécessaire
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialisation de l'application FastAPI
app = FastAPI()

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
        "iou_metric": iou_metric
    })
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None

# =========================================
# Palette et labels
# =========================================

PALETTE = [
    (0, 0, 0),       # Black for "Flat"
    (128, 0, 0),     # Red for "Human"
    (0, 128, 0),     # Green for "Vehicle"
    (128, 128, 0),   # Yellow for "Construction"
    (0, 0, 128),     # Blue for "Object"
    (128, 0, 128),   # Magenta for "Nature"
    (0, 128, 128),   # Cyan for "Sky"
    (128, 128, 128)  # Gray for "Void"
]

CLASS_LABELS = [
    "Flat",
    "Human",
    "Vehicle",
    "Construction",
    "Object",
    "Nature",
    "Sky",
    "Void"
]

# =========================================
# Routes
# =========================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Page d'accueil HTML intégrée.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>S.O.P.H.I.A - Accueil</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f9; text-align: center; padding: 20px; }
            h1 { color: #2c3e50; }
            a { font-size: 1.2rem; color: #ffffff; text-decoration: none; background-color: #3498db; padding: 10px 20px; border-radius: 5px; }
            a:hover { background-color: #2980b9; }
        </style>
    </head>
    <body>
        <h1>Bienvenue sur S.O.P.H.I.A</h1>
        <p>Bienvenue dans notre application de segmentation d'images.</p>
        <a href="/explore">Lister les images disponibles</a>
    </body>
    </html>
    """

@app.get("/explore", response_class=HTMLResponse)
async def explore():
    """
    Page HTML pour explorer les images disponibles.
    """
    base_image_dir = "api_image"
    cities = os.listdir(base_image_dir)

    if not cities:
        raise HTTPException(status_code=404, detail="Aucune ville trouvée dans le dossier des images.")

    city_options = "".join(f"<option value='{city}'>{city}</option>" for city in cities)

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Explorer les images</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; padding: 20px; }}
            select, button {{ font-size: 1rem; padding: 10px; margin: 10px; }}
        </style>
    </head>
    <body>
        <h1>Explorer les images</h1>
        <form id="explore-form" method="get" action="/view">
            <label for="city">Choisir une ville :</label>
            <select name="city" id="city" onchange="updateImages()" required>
                {city_options}
            </select>
            <label for="image_name">Choisir une image :</label>
            <select name="image_name" id="image_name" required>
                <option value="">Sélectionner une image</option>
            </select>
            <button type="submit">Visualiser</button>
        </form>

        <script>
            async function updateImages() {{
                const city = document.getElementById("city").value;
                const response = await fetch(`/list-images`);
                if (response.ok) {{
                    const data = await response.json();
                    const images = data[city]?.images || [];
                    const imageSelect = document.getElementById("image_name");
                    imageSelect.innerHTML = images.map(image => `<option value="${{image}}">${{image}}</option>`).join("");
                }} else {{
                    console.error("Erreur lors du chargement des images.");
                }}
            }}

            // Charger les images pour la ville sélectionnée au démarrage
            document.addEventListener("DOMContentLoaded", updateImages);
        </script>
    </body>
    </html>
    """

@app.get("/list-images", response_class=JSONResponse)
async def list_images():
    """
    Liste les images et masques disponibles par ville.
    """
    base_image_dir = "api_image"
    base_mask_dir = "api_mask"

    if not os.path.exists(base_image_dir) or not os.path.exists(base_mask_dir):
        raise HTTPException(status_code=404, detail="Les dossiers des images ou des masques sont introuvables.")

    # Initialisation du dictionnaire
    data = {}

    # Parcourir les villes dans le répertoire des images
    for city in os.listdir(base_image_dir):
        city_image_dir = os.path.join(base_image_dir, city)
        city_mask_dir = os.path.join(base_mask_dir, city)

        if not os.path.isdir(city_image_dir) or not os.path.isdir(city_mask_dir):
            continue

        # Liste des images pour la ville
        image_files = [
            f for f in os.listdir(city_image_dir)
            if f.endswith("_leftImg8bit.png")
        ]

        # Ajout des données pour la ville
        data[city] = {
            "images": image_files,
            "masks": [
                f.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                for f in image_files
            ]
        }

    # Retour des données JSON
    return data

@app.get("/view", response_class=HTMLResponse)
async def view(
    city: str,
    image_name: str,
    analyze: bool = Query(False),
    accept: str = Query("html")
):
    base_image_dir = "api_image"
    base_mask_dir = "api_mask"

    image_path = os.path.join(base_image_dir, city, image_name)
    mask_path = os.path.join(base_mask_dir, city, image_name.replace("leftImg8bit", "gtFine_labelIds"))

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        if accept == "json":
            return JSONResponse(status_code=404, content={"error": "Image ou masque non trouvé."})
        raise HTTPException(status_code=404, detail="Image ou masque non trouvé.")

    try:
        # Charger l'image et le masque annoté
        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # Redimensionner les images à 50 %
        img_resized = img.resize((img.width // 2, img.height // 2))
        mask_resized = mask.resize((mask.width // 2, mask.height // 2))

        # Encodage des images en base64
        img_base64 = encode_image_to_base64(img_resized)
        mask_base64 = encode_image_to_base64(mask_resized)

        # Initialisation du masque traité
        processed_mask_base64 = None
        if analyze:
            img_pred_resized = ImageOps.fit(img, (256, 256), Image.Resampling.LANCZOS)
            img_array = np.array(img_pred_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prédiction
            prediction = model.predict(img_array)[0]
            predicted_mask = np.argmax(prediction, axis=-1)

            # Générer le masque traité coloré
            predicted_mask_colored = Image.fromarray(apply_palette(predicted_mask, PALETTE))
            predicted_mask_resized = predicted_mask_colored.resize((img.width // 2, img.height // 2), resample=Image.NEAREST)

            processed_mask_base64 = encode_image_to_base64(predicted_mask_resized)

        # Générer le contenu JSON si demandé
        if accept == "json":
            return JSONResponse(content={
                "original_image": img_base64,
                "annotated_mask": mask_base64,
                "processed_mask": processed_mask_base64 if analyze else None,
                "legend": [{"label": label, "rgb": color} for label, color in zip(CLASS_LABELS, PALETTE)]
            })

        # Générer la légende des classes pour le HTML
        legend_html = "<div class='legend'>"
        for label, color in zip(CLASS_LABELS, PALETTE):
            rgb_color = f"rgb({color[0]}, {color[1]}, {color[2]})"
            legend_html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="width: 20px; height: 20px; background-color: {rgb_color}; margin-right: 10px;"></div>
                <span>{label}</span>
            </div>
            """
        legend_html += "</div>"

        # Ajouter le masque traité au HTML (si demandé)
        processed_mask_html = f"""
        <div>
            <h3>Masque traité</h3>
            <img src="data:image/jpeg;base64,{processed_mask_base64}" alt="Masque traité">
        </div>
        """ if analyze else ""

        # Retourner le HTML
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Visualisation</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; padding: 20px; }}
                img {{ max-width: 80%; margin: 10px; border: 1px solid #ccc; }}
                .container {{ display: flex; flex-direction: column; align-items: center; }}
                .button-container {{ margin-top: 20px; }}
                .legend {{ margin-top: 20px; text-align: left; display: inline-block; }}
            </style>
        </head>
        <body>
            <h1>Visualisation de l'image et des masques</h1>
            <div class="container">
                <div>
                    <h3>Image originale</h3>
                    <img src="data:image/jpeg;base64,{img_base64}" alt="Image originale">
                </div>
                <div>
                    <h3>Masque annoté</h3>
                    <img src="data:image/jpeg;base64,{mask_base64}" alt="Masque annoté">
                </div>
                {processed_mask_html}
                <div>
                    <h3>Légende des classes</h3>
                    {legend_html}
                </div>
            </div>
            <div class="button-container">
                <form action="/view" method="get">
                    <input type="hidden" name="city" value="{city}">
                    <input type="hidden" name="image_name" value="{image_name}">
                    <button type="submit" name="analyze" value="true">Analyser</button>
                </form>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la visualisation : {e}")



@app.get("/predict", response_class=HTMLResponse)
async def predict(city: str = Query(...), image_name: str = Query(...), accept: str = Query("html")):
    """
    Traite une image existante et retourne les résultats en HTML ou JSON.
    """
    # Traitement identique mais retour différencié JSON/HTML
    if accept == "html":
        # Retourner en HTML
        return HTMLResponse(content=f"""
            <html>
                <body>
                    <h1>Prédiction</h1>
                    <p>Image traitée : {image_name}</p>
                </body>
            </html>
        """)
    else:
        # Retourner en JSON
        return JSONResponse(content={"status": "success", "image_name": image_name})


# =========================================
# Utilitaires
# =========================================
def apply_palette(mask, palette):
    """
    Applique une palette de couleurs à un masque d'indices de classes.
    """
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        color_mask[mask == class_id] = color
    return color_mask

def encode_image_to_base64(image):
    """
    Encode une image en Base64.
    """
    buffered = tempfile.NamedTemporaryFile(delete=False)
    image.save(buffered, format="JPEG")
    with open(buffered.name, "rb") as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode("utf-8")
