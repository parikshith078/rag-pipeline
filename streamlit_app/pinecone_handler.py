from pinecone import Pinecone, ServerlessSpec
import os 
from dotenv import load_dotenv
import json
from emedding import generate_embeddings

# Load variables from the .env file
load_dotenv()

# Access them
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-pipeline-sources"

if index_name in pc.list_indexes().names():
    print("Using existing index")
else:
    print("Creating new index")
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc.Index(index_name)

def query_pinecone_db(query_text, k=5):
  query_embedding = generate_embeddings([query_text])[0].tolist()
  res = index.query(vector=query_embedding,top_k=k,include_values=True)
  res
  indices = [int(item["id"]) for item in res["matches"]]
  scores = [float(item["score"]) for item in res["matches"]]
  return indices,scores


def load_from_json(file_path: str) -> list[dict]:
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        print(f"Data successfully loaded from {file_path}")
        return data
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return []

source_file_path = "sources_and_chunks.json"
sources_and_chunks = load_from_json(source_file_path)
text_chunks = [item["sentence_chunk"] for item in sources_and_chunks]

def get_context_items(query_text):
    ids, scores = query_pinecone_db(query_text)
    contexts = [text_chunks[id] for id in ids]
    return contexts

def generate_prompt(query: str) -> str:
    # Retrieve context items
    context_items = get_context_items(query_text=query)

    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join(context_items)

    # Base prompt template
    base_prompt = f"""Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""
    
    return base_prompt