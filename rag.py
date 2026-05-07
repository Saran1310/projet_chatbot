import os
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq


# ── Constantes ─────────────────────────────────────────────────────────────────

FAISS_INDEX_PATH = "movies_index.faiss"
CSV_DATA_PATH    = "movies_data.csv"
EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL_NAME   = "llama-3.3-70b-versatile"
TOP_K            = 7
FALLBACK_K       = 50
MAX_RESULTS      = 5
SEUIL_CONFIANCE  = 0.45

LANGUES_DISPO = {
    "1": (None, "Tous les films (international)"),
    "2": ("fr", "Films en français 🇫🇷"),
    "3": ("en", "Films en anglais 🇬🇧"),
}

SYSTEM_PROMPT = """\
🎬 Hey ! Tu es un immense passionné de cinéma, le genre de personne capable de sortir \
une référence de film parfaite à n'importe quel moment 🍿😎

Ton objectif :
Aider l'utilisateur à trouver des films en utilisant UNIQUEMENT les films présents \
dans le contexte fourni.

Règles ultra importantes :
- Tu ne recommandes QUE des films présents dans le contexte.
- Pour chaque recommandation, tu mentionnes toujours :
  • le titre du film 🎥
  • sa note ⭐
- Si aucun film du contexte ne correspond à la demande, tu le dis clairement sans inventer.
- Si l'utilisateur demande un film très récent (après 2017) absent de la base, \
tu précises gentiment que ta base couvre les films jusqu'à 2017 ⏳
- Tu réponds toujours en français 🇫🇷

Style de réponse :
- Ton ton est cool, drôle, naturel et enthousiaste.
- Tu parles comme un vrai pote cinéphile qui adore partager ses découvertes.
- Tu peux faire des références à des films cultes, des scènes connues ou des répliques \
iconiques pour rendre les réponses plus fun 🎞️
- Tu peux utiliser quelques emojis avec modération.
- Même quand tu fais de l'humour ou des références, tu restes clair et utile.
"""

USER_PROMPT_TEMPLATE = """\
L'utilisateur a posé cette question : "{question}"

Voici les films les plus pertinents trouvés dans la base de données :
{context}
Réponds en français avec :
1. 🎬 Une recommandation pour chaque film pertinent (titre, année, note, résumé court)
2. 💡 Une explication du pourquoi ces films correspondent à la question
3. 🏆 Ton meilleur choix parmi ces films et pourquoi

Si aucun film ne correspond vraiment à la question, dis-le honnêtement.\
"""


# ── Initialisation ─────────────────────────────────────────────────────────────

def load_env_and_client() -> Groq:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY manquante dans le fichier .env !")
    return Groq(api_key=api_key)


def load_index_and_data() -> tuple[faiss.Index, pd.DataFrame]:

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CSV_DATA_PATH):
        raise FileNotFoundError(
            "Fichiers movies_index.faiss ou movies_data.csv introuvables.\n"
            "Lance d'abord : python indexation.py"
        )

    index = faiss.read_index(FAISS_INDEX_PATH)

    df = pd.read_csv(CSV_DATA_PATH)
    df["release_date"]       = df["release_date"].fillna("date inconnue")
    df["vote_average"]       = df["vote_average"].fillna(0)
    df["overview"]           = df["overview"].fillna("Pas de résumé disponible.")
    df["title"]              = df["title"].fillna("Titre inconnu")
    df["original_language"]  = df["original_language"].fillna("")

    return index, df


def init() -> tuple[Groq, SentenceTransformer, faiss.Index, pd.DataFrame]:
    """
    Initialise tous les composants nécessaires.
    Retourne (client_groq, embed_model, faiss_index, dataframe).
    """
    client      = load_env_and_client()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    index, df   = load_index_and_data()

    print(f"Index chargé : {index.ntotal} films disponibles")
    print(" Tapez votre question (ou 'quit' pour quitter)\n")

    return client, embed_model, index, df


# ── Saisie utilisateur ─────────────────────────────────────────────────────────

def choisir_langue() -> tuple[str | None, str]:

    print(" Filtrer les films par langue ?")
    for key, (_, label) in LANGUES_DISPO.items():
        print(f"  {key}. {label}")

    while True:
        choix = input("Votre choix : ").strip()
        if choix in LANGUES_DISPO:
            code, label = LANGUES_DISPO[choix]
            print(f" Filtre sélectionné : {label}\n")
            return code, label
        print(f"Entrez un chiffre parmi {list(LANGUES_DISPO.keys())}.")


# ── Recherche vectorielle ──────────────────────────────────────────────────────

def _encode_query(question: str, embed_model: SentenceTransformer) -> np.ndarray:
    """Encode la question et normalise le vecteur pour la recherche cosinus."""
    vector = embed_model.encode([question]).astype("float32")
    faiss.normalize_L2(vector)
    return vector


def _appliquer_filtre_langue(
    query_vector: np.ndarray,
    index: faiss.Index,
    df: pd.DataFrame,
    indices: np.ndarray,
    distances: np.ndarray,
    langue: str,
) -> tuple[pd.DataFrame, list[float]]:
    """
    Filtre les résultats par langue avec fallback si trop peu de résultats.
    Retourne (results_filtres, scores_alignes).
    """
    results = df.iloc[indices].copy()
    scores  = distances.tolist()

    # Aligner les scores avec les résultats AVANT filtrage
    results["_score"] = scores
    results = results[results["original_language"] == langue]

    if len(results) < 2:
        print(f" Peu de résultats en '{langue}', élargissement à {FALLBACK_K} films...")
        distances_large, indices_large = index.search(query_vector, FALLBACK_K)
        results = df.iloc[indices_large[0]].copy()
        results["_score"] = distances_large[0].tolist()
        results = results[results["original_language"] == langue]

    results = results.head(MAX_RESULTS).reset_index(drop=True)
    scores  = results["_score"].tolist()
    results = results.drop(columns=["_score"])

    return results, scores


def rechercher(
    question: str,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    df: pd.DataFrame,
    langue: str | None = None,
    k: int = TOP_K,
) -> tuple[pd.DataFrame, list[float]]:
    """Encode la question et retourne les films les plus proches avec leurs scores."""
    query_vector       = _encode_query(question, embed_model)
    distances, indices = index.search(query_vector, k)

    if langue:
        results, scores = _appliquer_filtre_langue(
            query_vector, index, df, indices[0], distances[0], langue
        )
    else:
        results = df.iloc[indices[0]].head(MAX_RESULTS).reset_index(drop=True)
        scores  = distances[0][:MAX_RESULTS].tolist()

    return results, scores


# ── Construction du contexte ───────────────────────────────────────────────────

def construire_contexte(
    results: pd.DataFrame,
    scores: list[float],
) -> tuple[str, list[str]]:
    """Formate les films trouvés en contexte lisible pour le LLM."""
    context = ""
    sources = []

    for i, (_, row) in enumerate(results.iterrows()):
        annee    = str(row["release_date"])[:4]
        score    = scores[i] if i < len(scores) else 0.0
        context += (
            f"Film {i+1} : {row['title']} ({annee}) "
            f"— Note : {row['vote_average']}/10 "
            f"[score similarité : {score:.2f}]\n"
            f"Résumé : {row['overview']}\n\n"
        )
        sources.append(f"{row['title']} ({annee}) — {row['vote_average']}/10")

    return context, sources


# ── Génération LLM ─────────────────────────────────────────────────────────────

def generer_reponse(question: str, context: str, client: Groq) -> str:
    prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── Affichage ─────────────────────────────────────────────────────────────────

def afficher_reponse(reponse: str, sources: list[str]) -> None:
    """Affiche la réponse du LLM et la liste des films analysés."""
    print("─" * 60)
    print(reponse)
    print("\n📚 Films analysés :")
    for i, source in enumerate(sources, 1):
        print(f"  {i}. {source}")
    print("─" * 60 + "\n")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    # Initialisation — aucune variable globale, tout est passé explicitement
    client, embed_model, index, df = init()

    # Choix de langue unique au démarrage
    langue, langue_label = choisir_langue()

    while True:
        question = input("🎬 Votre question : ").strip()

        if question.lower() in {"quit", "exit", "q"}:
            print("Au revoir ! 🎬")
            break

        if not question:
            print("⚠️  La question ne peut pas être vide.\n")
            continue

        print("\n⏳ Recherche en cours...\n")

        results, scores = rechercher(question, embed_model, index, df, langue=langue)

        if results.empty:
            print(
                f"⚠️  Aucun film trouvé avec le filtre '{langue_label}'.\n"
                "Relancez le programme sans filtre de langue si besoin.\n"
            )
            continue

        if scores[0] < SEUIL_CONFIANCE:
            print(
                f"⚠️  Peu de résultats pertinents (meilleur score : {scores[0]:.2f}).\n"
                "La réponse suivante peut être approximative.\n"
            )

        context, sources = construire_contexte(results, scores)

        try:
            reponse = generer_reponse(question, context, client)
            afficher_reponse(reponse, sources)
        except Exception as e:
            print(f"❌ Erreur lors de l'appel à Groq : {e}\n")


if __name__ == "__main__":
    main()
