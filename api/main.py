from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image, ImageOps
from io import BytesIO
import uvicorn

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
    (128, 128, 128), # Gray for "Void"
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

# =========================================
# Routes
# =========================================

@app.get("/", response_class=JSONResponse)
async def home():
    """
    Page d'accueil avec description des routes.
    """
    return {
        "message": "Bienvenue sur SOPHIA API.",
        "routes": {
            "/list-images": "Lister les images disponibles.",
            "/observe": "Visualiser une image et son masque annoté.",
            "/predict": "Envoyer une image pour prédire un masque.",
            "/visualize": "Combiner une image avec son masque prédictif."
        }
    }

@app.get("/list-images", response_class=JSONResponse)
async def list_images():
    """
    Liste les images et masques disponibles par ville.
    """
    base_image_dir = "api_image"
    base_mask_dir = "api_mask"

    if not os.path.exists(base_image_dir) or not os.path.exists(base_mask_dir):
        raise HTTPException(status_code=404, detail="Les dossiers api_image ou api_mask sont introuvables.")

    cities = os.listdir(base_image_dir)
    data = {}

    for city in cities:
        city_image_dir = os.path.join(base_image_dir, city)
        if os.path.isdir(city_image_dir):
            images = os.listdir(city_image_dir)
            masks = [img.replace("leftImg8bit", "gtFine_labelIds") for img in images]
            data[city] = {"images": images, "masks": masks}

    return data

@app.get("/observe")
async def observe(city: str, image_name: str):
    """
    Visualiser une image et son masque annoté.
    """
    base_image_dir = "api_image"
    base_mask_dir = "api_mask"

    image_path = os.path.join(base_image_dir, city, image_name)
    mask_path = os.path.join(base_mask_dir, city, image_name.replace("leftImg8bit", "gtFine_labelIds"))

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="L'image ou le masque annoté est introuvable.")

    return {
        "image": image_path,
        "mask": mask_path
    }

@app.post("/predict", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    """
    Traite une image et génère un masque prédit.
    """
    try:
        img = Image.open(file.file).convert("RGB")
        img_resized = ImageOps.fit(img, (256, 256), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        predicted_mask = np.argmax(prediction, axis=-1)

        combined_img = apply_palette(predicted_mask, PALETTE)
        mask_io = encode_image_to_io(combined_img)

        return FileResponse(mask_io, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")

@app.get("/visualize", response_class=FileResponse)
async def visualize(city: str, image_name: str):
    """
    Combine une image réelle avec son masque prédictif.
    """
    base_image_dir = "api_image"
    image_path = os.path.join(base_image_dir, city, image_name)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="L'image demandée est introuvable.")

    img = Image.open(image_path).convert("RGB")
    img_resized = ImageOps.fit(img, (256, 256), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_mask = np.argmax(prediction, axis=-1)

    combined_img = apply_palette(predicted_mask, PALETTE)
    combined_image_io = encode_image_to_io(combined_img)

    return FileResponse(combined_image_io, media_type="image/jpeg")

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

def encode_image_to_io(image_array):
    """
    Encode une image en BytesIO pour envoi HTTP.
    """
    img = Image.fromarray(image_array.astype(np.uint8))
    img_io = BytesIO()
    img.save(img_io, format="JPEG")
    img_io.seek(0)
    return img_io
