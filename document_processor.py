from pypdf import PdfReader
import re

def extract_text_from_pdf(pdf_path):
    """Extrait le texte complet d'un fichier PDF."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"  # Ajout d'un saut de ligne entre les pages
    return text

def split_into_chunks(text, max_sentences=3):
    """Découpe le texte en chunks basés sur un nombre maximum de phrases."""
    # Nettoyage initial : suppression des espaces inutiles
    text = re.sub(r'\s+', ' ', text).strip()

    # Découpage en phrases
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) == max_sentences:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    # Ajouter le dernier chunk s'il reste des phrases
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Exemple d'utilisation (pour tester le module)
if __name__ == '__main__':
    # ⚠️ IMPORTANT: Vous devez placer votre PDF dans le dossier du projet ou ajuster le chemin
    # Pour le test, créez un fichier 'test_doc.pdf' temporaire ou utilisez votre vrai fichier
    
    # Remplacez par le chemin de votre document PDF
    PDF_PATH = "votre_document_ia_generative.pdf" 
    
    try:
        full_text = extract_text_from_pdf(PDF_PATH)
        print(f"Texte extrait (début) : {full_text[:300]}...")
        
        chunks = split_into_chunks(full_text, max_sentences=3)
        print(f"\nNombre total de chunks créés : {len(chunks)}")
        print(f"Premier chunk pour vérification :\n---{chunks[0]}---")
        
    except FileNotFoundError:
        print(f"Erreur : Le fichier {PDF_PATH} est introuvable. Veuillez le placer dans le répertoire du projet.")