from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from rembg import remove
import easyocr
import numpy as np
import torch
import io
import webcolors

app = FastAPI()

# Carrega o modelo BLIP Large com suporte a prompts
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Remove o fundo branco da imagem
def remove_background(image_bytes):
    return remove(image_bytes)

def describe_image(image_bytes):
    raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def extract_text_from_image(image_bytes):
    reader = easyocr.Reader(['pt', 'en'], gpu=False)
    image_np = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    results = reader.readtext(np.array(image_np))
    ocr_text = " ".join([res[1] for res in results])
    return ocr_text

def extract_attributes(text):
    text_lower = text.lower()

    gender = "male" if "masculina" in text_lower else "female" if "feminina" in text_lower else "unisex"
    age_group = "adult" if "adulto" in text_lower else "kids" if "infantil" in text_lower else "adult"

    size_tags = ["pp", "p", "m", "g", "gg", "xg", "xxg"]
    size = next((s for s in size_tags if f" {s} " in f" {text_lower} "), "único")

    material = "algodão" if "algodão" in text_lower else "poliéster" if "poliéster" in text_lower else "desconhecido"

    return {
        "gender": gender,
        "age_group": age_group,
        "color": "desconhecida",  # A cor será preenchida mais tarde
        "size": size,
        "material": material
    }

# Função para encontrar a cor mais próxima da cor dominante usando webcolors
def closest_color(requested_color):
    """ Encontra o nome da cor mais próxima para o valor RGB fornecido. """
    try:
        # Tenta obter o nome exato
        return webcolors.rgb_to_name(requested_color, spec='css3')
    except ValueError:
        # Se não encontrar, calcula a cor mais próxima
        min_distance = float('inf')
        closest_name = None

        for name in webcolors.names("css3"):
            hex_value = webcolors.name_to_hex(name)
            color_rgb = webcolors.hex_to_rgb(hex_value)
            distance = np.linalg.norm(np.array(requested_color) - np.array(color_rgb))

            if distance < min_distance:
                min_distance = distance
                closest_name = name

        return closest_name or webcolors.rgb_to_hex(requested_color)

# Função para obter a cor dominante da imagem
def get_dominant_color(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((50, 50))
    pixels = np.array(image)
    pixels = pixels.reshape((-1, 3))

    avg_color = pixels.mean(axis=0)
    dominant_color = tuple(map(int, avg_color))

    return dominant_color

@app.post("/extract-attributes")
async def extract(image: UploadFile, title: str = Form(...), description: str = Form(...)):
    original_bytes = await image.read()
    image_no_bg_bytes = remove_background(original_bytes)

    # BLIP Caption com imagem sem fundo
    caption = describe_image(image_no_bg_bytes)

    # OCR com imagem sem fundo
    ocr_text = extract_text_from_image(image_no_bg_bytes)

    # Cor dominante da imagem sem fundo
    dominant_color = get_dominant_color(image_no_bg_bytes)
    color_name = closest_color(dominant_color)

    full_text = f"{caption}. {ocr_text}. {title}. {description}"
    attributes = extract_attributes(full_text)
    attributes["color"] = color_name

    return JSONResponse(content={
        "caption": caption,
        "ocr_text": ocr_text,
        "dominant_color": dominant_color,
        "attributes": attributes
    })
