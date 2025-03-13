# modules/vision_classifier.py

import base64
import requests
import json

def classify_image_with_llama(image_path: str) -> dict:
    """
    Stuur een afbeelding naar het LLaMA 3.2 90B Visio Preview-model (via Groqcloud),
    en vraag om een JSON-classificatie van brand, color, condition, quality, etc.

    :param image_path: Pad naar de lokale afbeelding (jpg/png/etc.)
    :return: Dict met beschrijving en categorieën (brand, color, condition, quality).
    """

    # Stap 1: Lees de afbeelding en converteer naar base64 (als voorbeeld)
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Stap 2: Stel de prompt samen
    # Voorbeeldprompt zoals aangegeven in je vraag:
    prompt = (
        "You are image describer. First you describe the image in clear detail. "
        "Then you try to describe this in categories. \n\n"
        "Put the categories in JSON and try to find the brand, color, condition, "
        "and quality of the objects in the image."
    )

    # Stap 3: Stel de payload voor de API-call samen
    # Let op: Dit is voorbeeld/pseudo-code; je zult de echte Groqcloud endpoint en parameters
    # moeten invullen, net zoals de authentificatie headers, etc.
    payload = {
        "model": "llama-3.2-90b-vision-preview",
        "prompt": prompt,
        "image_b64": image_b64,  # Of soms "image": image_b64
        "max_tokens": 300,
        "temperature": 0.3
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer <YOUR_GROQCLOUD_API_KEY>"
    }

    # Stap 4: Doe de request naar de Groqcloud API
    # Vervang 'https://api.groqcloud.example/v1/vision' door de echte URL
    response = requests.post(
        "https://api.groqcloud.example/v1/vision",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        data = response.json()
        # Stel dat de API het modelantwoord in 'generated_text' zet.
        # Het modelantwoord zou een (grotendeels) JSON-achtige string kunnen zijn,
        # eventueel met extra tekst eromheen.
        model_answer = data.get("generated_text", "")
        # Probeer het JSON-gedeelte eruit te halen
        # Als het model exact JSON teruggeeft, kun je direct inladen met `json.loads`
        try:
            # (Eenvoudig voorbeeld, kan nodig zijn om extra cleanup te doen)
            parsed_json = json.loads(extract_json_part(model_answer))
            return parsed_json
        except json.JSONDecodeError:
            # Als het niet lukt te parsen, geef een fallback
            print("Kon geen geldige JSON uit het modelresultaat parsen.")
            return {"error": "Invalid JSON from model", "raw_answer": model_answer}

    else:
        print(f"Fout bij API-call: status code {response.status_code} - {response.text}")
        return {"error": f"API error: {response.status_code}", "details": response.text}


def extract_json_part(text: str) -> str:
    """
    Hulpfunctie die probeert de eerste geldige JSON in een tekst te extraheren.
    Heel simpel voorbeeld: zoekt naar eerste '{' tot bijpassende '}'.

    In een echte toepassing zou je hier een robuustere parser kunnen gebruiken
    (regex, of langchain's re.json parse, etc.).
    """
    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_str = text[start_index:end_index + 1]
        return json_str
    else:
        return text  # Fallback: hele text
