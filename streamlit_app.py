import streamlit as st
import os

# Important : Nous avons besoin que les fonctions de chargement de modèles (@st.cache_resource) 
# de rag_pipeline.py soient définies AVANT d'être appelées. 
# Pour un déploiement propre, on place l'import ici pour utiliser la mise en cache de Streamlit.

# Assurez-vous d'avoir les dépendances Streamlit, rag_pipeline et document_processor
from rag_pipeline import FaissIndexManager, ConversationalRAG, get_index_manager, load_models, configure_gemini, EMBEDDING_MODEL_NAME
from document_processor import extract_text_from_pdf

# --- Configuration Streamlit ---
st.set_page_config(page_title="🤖 Chatbot RAG sur IA Générative", layout="wide")
st.title("🤖 Chatbot RAG : Conception de Produits Augmentée par IA Générative")

# --- Configuration de l'API Gemini ---
# Vérifier si la clé API existe dans un fichier .env ou comme variable d'environnement
API_KEY_FILE = ".gemini_key"
if "gemini_configured" not in st.session_state:
    st.session_state["gemini_configured"] = False
    
# Charger la clé depuis le fichier s'il existe
if os.path.exists(API_KEY_FILE) and not st.session_state["gemini_configured"]:
    with open(API_KEY_FILE, "r") as f:
        saved_key = f.read().strip()
        if saved_key:
            try:
                gemini_model = configure_gemini(saved_key)
                st.session_state["gemini_model"] = gemini_model
                st.session_state["gemini_configured"] = True
            except:
                pass

if not st.session_state["gemini_configured"]:
    st.warning("⚠️ Veuillez configurer votre clé API Google Gemini pour commencer")
    api_key = st.text_input("Clé API Gemini:", type="password", help="Obtenez votre clé gratuite sur https://makersuite.google.com/app/apikey")
    
    col1, col2 = st.columns(2)
    with col1:
        save_key = st.checkbox("Sauvegarder la clé pour les prochaines sessions", value=True)
    
    if st.button("Configurer"):
        if api_key:
            try:
                gemini_model = configure_gemini(api_key)
                st.session_state["gemini_model"] = gemini_model
                st.session_state["gemini_configured"] = True
                
                # Sauvegarder la clé si demandé
                if save_key:
                    with open(API_KEY_FILE, "w") as f:
                        f.write(api_key)
                    st.success("✅ API Gemini configurée et clé sauvegardée avec succès!")
                else:
                    st.success("✅ API Gemini configurée avec succès!")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur de configuration: {e}")
        else:
            st.error("Veuillez entrer une clé API valide")
    st.stop()

# Charger les modèles (mis en cache par @st.cache_resource dans rag_pipeline.py)
EMBEDDER = load_models()
INDEX_MANAGER = get_index_manager()
CHATBOT = ConversationalRAG(index_manager=INDEX_MANAGER, embedder=EMBEDDER, gemini_model=st.session_state["gemini_model"])

# --- Configuration Streamlit ---
st.set_page_config(page_title="🤖 Chatbot RAG sur IA Générative", layout="wide")
st.title("🤖 Chatbot RAG : Conception de Produits Augmentée par IA Générative")

# --- Initialisation de l'état de la session ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "pdf_path" not in st.session_state:
    st.session_state["pdf_path"] = None

if "index_initialized" not in st.session_state:
    st.session_state["index_initialized"] = False
    
# ⚠️ REMPLACEZ PAR LE NOM DE VOTRE FICHIER PDF
DEFAULT_PDF_NAME = "IA_Générative_pour_Conception_Produit.pdf" 

# --- Fonction d'Initialisation de la Base Documentaire ---
def initialize_index():
    if not st.session_state["index_initialized"] and os.path.exists(DEFAULT_PDF_NAME):
        with st.spinner(f"Initialisation de l'index FAISS avec {DEFAULT_PDF_NAME}..."):
            try:
                num_chunks = INDEX_MANAGER.add_documents(DEFAULT_PDF_NAME, "Base Initiale")
                st.session_state["index_initialized"] = True
                st.success(f"Indexation terminée : {num_chunks} chunks ajoutés à l'index.")
            except Exception as e:
                st.error(f"Erreur lors de l'indexation initiale : {e}")

# Appel à l'initialisation au démarrage de l'application
initialize_index()

# --- Barre Latérale (Upload et Statistiques) ---
with st.sidebar:
    st.header("Gestion de la Base de Connaissances")
    
    # Upload dynamique de document
    uploaded_file = st.file_uploader(
        "Ajouter un nouveau document (.pdf) à la base de connaissances :", 
        type=["pdf"]
    )

    if uploaded_file is not None:
        if st.button("Indexer le nouveau document"):
            # Sauvegarder le fichier temporairement
            temp_path = os.path.join("./", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.spinner(f"Indexation du document {uploaded_file.name}..."):
                try:
                    num_chunks = INDEX_MANAGER.add_documents(temp_path, uploaded_file.name)
                    st.success(f"'{uploaded_file.name}' indexé : {num_chunks} chunks ajoutés. Total chunks: {INDEX_MANAGER.index.ntotal}")
                except Exception as e:
                    st.error(f"Erreur d'indexation pour le nouveau document : {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path) # Nettoyage

    st.subheader("Statistiques RAG")
    st.info(f"Modèle d'Embeddings : {EMBEDDING_MODEL_NAME}\n"
            f"Modèle Génératif : Google Gemini 2.5 Flash\n"
            f"Total Chunks Indexés : {INDEX_MANAGER.index.ntotal}")
    
    if st.button("Réinitialiser la conversation"):
        st.session_state["messages"] = []
        st.rerun()

# --- Logique de la Conversation ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("retrieved_chunks"):
            with st.expander("📚 Chunks utilisés (Traçabilité)"):
                for chunk in message["retrieved_chunks"]:
                    st.caption(f"**Source:** {chunk['source']} | **Score Cosinus:** {chunk['score']:.4f}")
                    st.text(chunk["content"])

if prompt := st.chat_input("Posez votre question sur la conception de produits augmentée par l'IA générative..."):
    # 1. Ajouter le message utilisateur à l'historique Streamlit
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Afficher la question
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Générer la réponse RAG
    with st.chat_message("assistant"):
        with st.spinner("Recherche de passages et génération de la réponse..."):
            try:
                # Construire l'historique au format attendu par ConversationalRAG
                history = [(st.session_state.messages[i]["content"], st.session_state.messages[i+1]["content"]) 
                           for i in range(0, len(st.session_state.messages)-1, 2)
                           if i+1 < len(st.session_state.messages) and st.session_state.messages[i+1]["role"] == "assistant"]
                
                # Récupérer les chunks pertinents
                retrieved_chunks = INDEX_MANAGER.search(prompt, top_k=3)
                
                # Générer la réponse
                answer = CHATBOT.answer_question(prompt, history)
                
                # Afficher la réponse
                st.markdown(answer)
                
                # Afficher les chunks utilisés pour la traçabilité
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "retrieved_chunks": retrieved_chunks
                })
            except Exception as e:
                error_msg = f"Erreur lors de la génération de la réponse : {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })