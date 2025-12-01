import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from document_processor import extract_text_from_pdf, split_into_chunks
import streamlit as st
import google.generativeai as genai

# --- Configuration et Modèles Globaux ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialisation des modèles
@st.cache_resource
def load_models():
    """Charge le modèle d'embeddings une seule fois."""
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME).to(DEVICE)
    return embedder

EMBEDDER = load_models()
EMBEDDING_DIM = 384 # Dimension de all-MiniLM-L6-v2

# Configuration Gemini
def configure_gemini(api_key):
    """Configure l'API Gemini avec la clé fournie."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    return model

# --- Gestion de l'Index Vectoriel FAISS ---
class FaissIndexManager:
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        # Index FlatIP (Inner Product) pour la similarité cosinus (si embeddings normalisés)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks = []
        self.doc_name = "Base Initiale"

    def add_documents(self, pdf_path, doc_name):
        """Charge un PDF, le découpe, l'encode et met à jour l'index FAISS."""
        full_text = extract_text_from_pdf(pdf_path)
        new_chunks = split_into_chunks(full_text)
        
        # Encoder les nouveaux chunks
        new_embeddings = EMBEDDER.encode(
            new_chunks, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        ).astype('float32') # FAISS nécessite float32

        # Ajouter à l'index et à la liste des chunks
        self.index.add(new_embeddings)
        self.chunks.extend(new_chunks)
        self.doc_name = doc_name # Met à jour le nom du document principal

        return len(new_chunks)

    def search(self, query, top_k=3):
        """Recherche les top_k chunks les plus pertinents pour une requête."""
        if self.index.ntotal == 0:
            return []

        # Encoder la requête
        query_emb = EMBEDDER.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        ).astype('float32')
        
        # Recherche FAISS (D = distances/scores, I = indices)
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "score": float(score),
                    "content": self.chunks[int(idx)],
                    "source": self.doc_name # Ajouter la source pour la traçabilité
                })
        return results

# --- Logique de Chatbot Conversationnel RAG ---
class ConversationalRAG:
    """
    Classe pour orchestrer le pipeline Retrieval-Augmented Generation (RAG).
    """
    def __init__(self, index_manager, embedder, gemini_model):
        """
        Initialise le pipeline RAG.

        Args:
            index_manager (FaissIndexManager): Gestionnaire de l'index FAISS.
            embedder (SentenceTransformer): Modèle d'embeddings.
            gemini_model: Modèle Gemini pour la génération.
        """
        self.index_manager = index_manager
        self.embedder = embedder
        self.gemini_model = gemini_model

    def build_chat_prompt(self, history, question, retrieved_chunks):
        """
        Construit le prompt pour le modèle génératif.

        Args:
            history (list): Historique de la conversation (liste de tuples utilisateur/assistant).
            question (str): Question posée par l'utilisateur.
            retrieved_chunks (list): Chunks récupérés depuis l'index FAISS.

        Returns:
            str: Prompt formaté pour le modèle génératif.
        """
        # Nettoyer et formatter les chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks[:3]):
            clean_chunk = chunk.strip().replace('\n', ' ').replace('  ', ' ')
            context_parts.append(f"[Document {i+1}]: {clean_chunk}")
        
        context = "\n\n".join(context_parts)
        
        # Prompt optimisé pour Gemini
        prompt = f"""Tu es un assistant spécialisé en IA générative pour la conception de produits. Réponds UNIQUEMENT en français et en te basant sur les documents fournis ci-dessous.

DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Réponds en français de manière claire, précise et complète
- Base-toi UNIQUEMENT sur les informations des documents ci-dessus
- Si l'information n'est pas dans les documents, dis "Je ne dispose pas de cette information dans les documents fournis"
- Structure ta réponse de manière claire avec des paragraphes si nécessaire

RÉPONSE:"""
        return prompt

    def generate_response(self, prompt):
        """
        Génère une réponse à partir du prompt donné via Gemini.

        Args:
            prompt (str): Prompt formaté pour le modèle génératif.

        Returns:
            str: Réponse générée par le modèle.
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Erreur lors de la génération: {str(e)}"

    def answer_question(self, question, history):
        """
        Répond à une question en utilisant le pipeline RAG.

        Args:
            question (str): Question posée par l'utilisateur.
            history (list): Historique de la conversation.

        Returns:
            str: Réponse générée par le pipeline.
        """
        # Récupérer les chunks pertinents
        retrieved_chunks_data = self.index_manager.search(question, top_k=3)
        retrieved_chunks = [chunk["content"] for chunk in retrieved_chunks_data]

        # Construire le prompt
        prompt = self.build_chat_prompt(history, question, retrieved_chunks)

        # Générer la réponse
        return self.generate_response(prompt)

# --- Initialisation FAISS Manager ---
@st.cache_resource
def get_index_manager():
    """Instancie et met en cache le gestionnaire d'index pour Streamlit."""
    return FaissIndexManager()

def search_faiss(query, index, embedder, top_k=3):
    """
    Interroge l'index FAISS pour récupérer les chunks les plus pertinents.

    Args:
        query (str): La requête utilisateur.
        index (faiss.Index): L'index FAISS.
        embedder (SentenceTransformer): Le modèle d'embeddings.
        top_k (int): Nombre de résultats les plus pertinents à retourner.

    Returns:
        list: Liste des indices et scores des chunks les plus pertinents.
    """
    # Encoder la requête utilisateur en embedding
    query_embedding = embedder.encode([query], normalize_embeddings=True).astype('float32')

    # Rechercher les top_k résultats dans l'index FAISS
    distances, indices = index.search(query_embedding, top_k)

    return indices[0], distances[0]