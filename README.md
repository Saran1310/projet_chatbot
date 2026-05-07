# RAG - Recommandation de Films

Projet réalisé dans le cadre du cours RAG & LLM.
On a choisi le sujet A : construire un assistant de recommandation de films à partir du dataset TMDB 5000.

# Sujet choisi
Sujet A — Recommandation de Films  
Dataset : [TMDB 5000 Movie Dataset (Kaggle)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

# Comment ça marche
Le système fonctionne en deux étapes :
Étape 1 - Indexation (à faire une seule fois)  
On charge le CSV, on nettoie les données, on construit un texte par film (titre + genres + synopsis), on génère les embeddings et on sauvegarde l'index FAISS sur disque.

Étape 2 - Questions-réponses (à chaque lancement)  
On recharge l'index, l'utilisateur pose une question, on cherche les films les plus proches dans FAISS, on envoie le contexte au LLM Groq et il génère une recommandation.

CSV → nettoyage → embeddings → index FAISS (sauvegardé)
   ↓
question → embedding → recherche → contexte → Groq → réponse

# Structure du projet
.
├── Data/
│   └── tmdb_5000_movies.csv
├── indexation.py
├── rag_movies_v3.py
├── movies_index.faiss          
├── movies_data.csv             
├── requirements.txt
├── .env                        
├── .gitignore
└── README.md

# Installation

bash
-Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate        

-Installer les dépendances
pip install -r requirements.txt

Créer un fichier `.env` à la racine :

GROQ_API_KEY= la_clé_api

Clé gratuite sur [console.groq.com](https://console.groq.com).

Téléchargement de `tmdb_5000_movies.csv` sur Kaggle et le placer dans `Data/`.

# Lancement
bash
 1. Indexation 
python indexation.py

 2. Lancer l'assistant
python rag_movies_v3.py

Au démarrage on choisit un filtre de langue (tous / français / anglais), ensuite on peut poser des questions librement. 
`quit` pour quitter.

# Nos choix techniques
Pourquoi `paraphrase-multilingual-mpnet-base-v2` ? 
Les synopsis du dataset sont en anglais mais on voulait pouvoir poser des questions en français. Ce modèle multilingue gère les deux, donc c'était le choix logique.

Pourquoi titre + genres + synopsis dans l'embedding ?
On a testé avec seulement le synopsis au début, mais les résultats pour des requêtes comme "film d'action" étaient moins bons. Ajouter les genres dans le texte a clairement amélioré ça. Le titre permet aussi de retrouver un film par son nom directement.

Pourquoi `IndexFlatIP` avec normalisation L2 et pas `IndexFlatL2` ? 
`IndexFlatIP` avec des vecteurs normalisés revient à calculer la similarité cosinus, ce qui compare le sens des textes indépendamment de leur longueur. C'est plus adapté que la distance euclidienne pour de la recherche sémantique.

Persistance de l'index 
On sauvegarde l'index avec `faiss.write_index()` et les données avec `df.to_csv()` dans le même ordre. Comme ça on n'a pas besoin de réindexer à chaque lancement, ça prend 10 secondes au lieu de 10 minutes.

Filtre par langue 
On filtre les résultats par langue au démarrage. Si le filtre retourne moins de 2 films, on élargit automatiquement la recherche à 50 candidats pour éviter de se retrouver avec un contexte vide.

Score de confiance  
Si le meilleur score FAISS est en dessous de 0.45, on prévient l'utilisateur que les résultats sont peut-être approximatifs. C'est le bonus B du TP.

# Réponses aux questions du TP
Q1 - Conversion CSV → texte  
On concatène titre + genres + synopsis. Les autres colonnes (note, date, durée) restent en métadonnées, elles ne servent qu'à l'affichage.

Q2 - Extraction des genres JSON  
La colonne `genres` est au format `[{"id": 18, "name": "Drama"}, ...]`. On utilise `json.loads()` pour parser et on récupère uniquement les champs `name`.

Q3 - Persistance 
`faiss.write_index()` pour l'index, `df.to_csv()` pour les métadonnées.Les deux doivent être dans le même ordre pour que les indices retournés par FAISS correspondent aux bons films.

Q4 - Guider le LLM 
Le prompt système lui impose de ne recommander que les films du contexte, de toujours citer le titre et la note, et de ne pas inventer si rien ne correspond.

Q5 - Films récents 
Le dataset s'arrête à 2017. Si quelqu'un demande un film de 2023, le LLM est instruit de le signaler plutôt que d'halluciner.

# Limites

- Le dataset couvre jusqu'à 2017 seulement.
- Les synopsis sont en anglais, ce qui peut légèrement dégrader la précision sur des questions très spécifiques en français.
- Le filtre de langue se choisit au démarrage, pas question par question.
- Pas de reranker : on s'appuie uniquement sur la similarité cosinus, ce qui reste une approximation.

# Dépendances
groq
sentence-transformers
faiss-cpu
pandas
python-dotenv
requests

AWOUTE Améyo Grace 
DIABAGATE NA-MAMAN Saran
