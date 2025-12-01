from rag_pipeline import load_models, FaissIndexManager, ConversationalRAG

# Charger les modèles
embedder, tokenizer, gen_model = load_models()

# Initialiser l'index FAISS
index_manager = FaissIndexManager()
index_manager.add_documents("IA_Générative_pour_Conception_Produit.pdf", "Base Initiale")

# Tester la classe ConversationalRAG
chatbot = ConversationalRAG(index_manager, embedder, tokenizer, gen_model)
history = [("Qu'est-ce que l'IA ?", "L'IA est la science des machines intelligentes.")]
question = "Quels sont les principes éthiques de l'IA ?"
response = chatbot.answer_question(question, history)

print("Réponse générée :", response)