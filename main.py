from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Charger le modèle et le tokenizer
MODEL_NAME = "Helsinki-NLP/opus-mt-fr-en"  # Changez par votre modèle "NeNeAI_007_Model" si déjà sauvegardé
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Initialisation de l'application FastAPI
app = FastAPI(title="API de traduction Français → Kiluba", version="1.0")

# Schéma de données pour les requêtes
class TranslationRequest(BaseModel):
    text: str

# Endpoint principal pour la traduction
@app.post("/translate/")
async def translate_text(request: TranslationRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Le texte d'entrée ne peut pas être vide.")
    
    try:
        # Traduction du texte
        inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"translation": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")
