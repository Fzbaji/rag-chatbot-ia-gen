import google.generativeai as genai

# Configuration avec votre clé API
API_KEY = "----------------------------"
genai.configure(api_key=API_KEY)

# Test du modèle Gemini
model = genai.GenerativeModel('gemini-2.5-flash')

# Test simple
response = model.generate_content("Qu'est-ce que l'optimisation topologique en conception de produits ? Réponds en français.")
print("✅ Test Gemini réussi!")
print("\nRéponse:")
print(response.text)
