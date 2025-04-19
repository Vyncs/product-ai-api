from PIL import Image
import requests
from io import BytesIO

def analyze_image(image_url: str) -> dict:
    # Baixa a imagem da URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # 👇 Simulação de IA — depois vamos trocar por modelo real
    width, height = image.size
    color = "cinza" if "cinza" in image_url.lower() else "preto"

    # Apenas mocks por enquanto
    return {
        "color": color,
        "size": "desconhecido",
        "age_group": "adult",
        "gender": "unisex"
    }
