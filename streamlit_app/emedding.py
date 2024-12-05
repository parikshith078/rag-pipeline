import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device=device)
def generate_embeddings(text_list):
  embeddings = embedding_model.encode(text_list, convert_to_tensor=True)
  return embeddings