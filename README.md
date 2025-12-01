# 🤖 Mini-Chatbot RAG : Conception de Produits Augmentée par IA Générative

## 📋 Description du Projet

Ce projet implémente un **chatbot conversationnel basé sur RAG (Retrieval-Augmented Generation)** spécialisé dans le domaine de l'IA générative pour la conception de produits. Le système permet d'interroger intelligemment une base de connaissances documentaire et d'obtenir des réponses contextuelles précises en français.

## 🎯 Objectifs

1. **Répondre à des questions spécialisées** : Le chatbot répond aux questions sur l'IA générative, l'optimisation topologique, le DFMA, et d'autres concepts liés à la conception de produits
2. **Indexation dynamique** : Possibilité d'ajouter de nouveaux documents PDF à la base de connaissances en temps réel
3. **Performance et légèreté** : Architecture optimisée pour fonctionner sur CPU sans nécessiter de GPU puissant
4. **Traçabilité** : Affichage des sources (chunks) utilisées pour générer chaque réponse

## 🏗️ Architecture du Système

Le projet est organisé en 3 composants principaux :

### 1. **Traitement de Documents** (`document_processor.py`)
- **Extraction de texte** depuis des fichiers PDF (via `pypdf`)
- **Chunking intelligent** : Découpage du texte en segments de 3 phrases pour maintenir la cohérence sémantique
- Support des documents texte et PDF

### 2. **Pipeline RAG** (`rag_pipeline.py`)
- **Modèle d'embeddings** : `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
  - Convertit le texte en vecteurs numériques pour la recherche sémantique
- **Index vectoriel FAISS** : Stockage et recherche rapide des embeddings
  - Utilise `IndexFlatIP` (Inner Product) pour la similarité cosinus
- **Modèle génératif** : Google Gemini 2.5 Flash via API
  - Génération de réponses de haute qualité en français
  - Alternative performante aux modèles T5 pour le multilingue
- **Classe ConversationalRAG** : Orchestration complète du pipeline
  - Retrieval : Récupération des 3 chunks les plus pertinents
  - Prompt construction : Construction d'un prompt optimisé avec contexte
  - Generation : Génération de la réponse finale

### 3. **Interface Utilisateur** (`streamlit_app.py`)
- **Interface web interactive** avec Streamlit
- **Gestion de session** : Historique de conversation persistant
- **Upload dynamique** : Ajout de nouveaux documents via interface
- **Configuration API** : Sauvegarde automatique de la clé API Gemini
- **Affichage des sources** : Traçabilité des chunks utilisés pour chaque réponse

## 🛠️ Technologies Utilisées

### Frameworks & Bibliothèques
- **Streamlit** : Interface web interactive
- **Transformers** : Infrastructure pour les modèles de NLP
- **Sentence-Transformers** : Modèles d'embeddings sémantiques
- **FAISS** : Recherche vectorielle ultra-rapide (Facebook AI)
- **PyPDF** : Extraction de texte depuis PDF
- **Google Generative AI** : API Gemini pour la génération de texte

### Modèles IA
- **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2`
  - Léger (80 MB)
  - Performant pour la recherche sémantique
  - Multilingue
- **Génération** : Google Gemini 2.5 Flash
  - Excellente qualité en français
  - Gratuit jusqu'à 60 requêtes/minute
  - API simple et rapide

### Outils
- **Python 3.10+**
- **PyTorch** : Backend pour les modèles
- **NumPy** : Manipulation des vecteurs

## 📦 Installation

### Prérequis
- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner le projet**
```bash
git clone <url-du-repo>
cd rag-chatbot-ia-gen
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
```

3. **Activer l'environnement virtuel**
- Windows PowerShell :
```powershell
.\venv\Scripts\Activate.ps1
```
- Linux/Mac :
```bash
source venv/bin/activate
```

4. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

5. **Obtenir une clé API Google Gemini**
- Rendez-vous sur https://makersuite.google.com/app/apikey
- Créez une clé API gratuite
- Copiez la clé (elle sera demandée au premier lancement)

## 🚀 Utilisation

### Lancer l'application

```bash
streamlit run streamlit_app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse `http://localhost:8501`.

### Premier lancement

1. **Configuration de l'API** :
   - Entrez votre clé API Google Gemini
   - Cochez "Sauvegarder la clé pour les prochaines sessions" (recommandé)
   - Cliquez sur "Configurer"

2. **Indexation initiale** :
   - L'application indexe automatiquement le document `IA_Générative_pour_Conception_Produit.pdf`
   - Attendez la fin de l'indexation (quelques secondes)

3. **Utilisation du chatbot** :
   - Posez vos questions dans la zone de texte en bas
   - Le système récupère les passages pertinents et génère une réponse
   - Cliquez sur "📚 Chunks utilisés (Traçabilité)" pour voir les sources

### Ajouter de nouveaux documents

1. Utilisez le **file uploader** dans la barre latérale
2. Sélectionnez un fichier PDF
3. Cliquez sur "Indexer le nouveau document"
4. Le document est ajouté à la base de connaissances en temps réel


## 🔧 Configuration Avancée

### Paramètres du Pipeline RAG

Dans `rag_pipeline.py`, vous pouvez ajuster :

- **Dimension des embeddings** : `EMBEDDING_DIM = 384`
- **Nombre de chunks récupérés** : `top_k=3` dans la méthode `search()`
- **Modèle d'embeddings** : Modifier `sentence-transformers/all-MiniLM-L6-v2`

### Paramètres du Chunking

Dans `document_processor.py` :

- **Taille des chunks** : `max_sentences=3` (actuellement 3 phrases par chunk)

### Paramètres de Génération

Dans `rag_pipeline.py`, méthode `generate_response()` :

- **Longueur des réponses** : Ajuster les paramètres du modèle Gemini si nécessaire

## 📊 Fonctionnalités

### ✅ Implémentées
- [x] Extraction de texte depuis PDF
- [x] Chunking par phrases (3 phrases/chunk)
- [x] Embeddings avec Sentence-Transformers
- [x] Index vectoriel FAISS
- [x] Recherche sémantique
- [x] Génération de réponses avec Gemini
- [x] Interface Streamlit interactive
- [x] Upload dynamique de documents
- [x] Historique de conversation
- [x] Traçabilité des sources
- [x] Sauvegarde de la clé API
- [x] Réponses en français de haute qualité


