import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np 
import pandas as pd
import json

df=pd.read_csv("Data/tmdb_5000_movies.csv")
print(df['genres'])

def extract_val_genres(genre_init):
    try:
        genres = json.loads(genre_init)
        return " ".join([i["name"] for i in genres])
    except:
        return ""
    
df["new_genres"] = df["genres"].apply(extract_val_genres)

df["new_genres"] = df["genres"].apply(extract_val_genres)
df["overview"] = df["overview"].fillna("")  
df["title"] = df["title"].fillna("")

df["text"] = df["title"] + " " + df["new_genres"] + " " + df["overview"]
    
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

embeddings = model.encode(
    df["text"].tolist(),
    show_progress_bar=True,
    batch_size=64  
)


embeddings = np.array(embeddings).astype("float32")

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)


print(f"Index créé avec {index.ntotal} films.")


faiss.write_index(index, "movies_index.faiss")
df.to_csv("movies_data.csv", index=False)
print("Sauvegarde terminée !")