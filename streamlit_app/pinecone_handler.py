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
    Query: What are the safety guidelines for handling Machine 23 in a chemical plant?
    Answer: Safety guidelines for handling Machine 23 in a chemical plant include several precautions to ensure operator safety and minimize the risk of accidents. These guidelines typically include wearing personal protective equipment (PPE) such as gloves, goggles, and flame-resistant clothing. Operators should also be trained in emergency shutdown procedures in case of equipment malfunction. It's important to regularly inspect Machine 23 for signs of wear or potential failure. Proper lockout/tagout (LOTO) procedures should be followed to ensure that the machine is de-energized during maintenance activities. Additionally, operators must be aware of the specific hazards associated with the chemicals being processed, such as flammability, toxicity, or corrosiveness.

    \nExample 2:
    Query: How can an MES help improve inventory management in a chemical manufacturing facility?
    Answer: A Manufacturing Execution System (MES) can significantly improve inventory management in a chemical manufacturing facility by providing real-time tracking of raw materials, intermediate products, and finished goods. Through integration with sensors and automated systems, MES allows for accurate tracking of inventory levels, usage rates, and production status. This real-time data helps prevent stockouts or overstocking by providing insights into material demand and supply. Additionally, MES can generate automatic alerts for reordering materials, reducing the chances of human error and improving overall efficiency in the supply chain.

    \nExample 3:
    Query: What are the common challenges when implementing an MES in a chemical plant?
    Answer: Common challenges when implementing a Manufacturing Execution System (MES) in a chemical plant include system integration, data accuracy, and employee training. Integrating MES with existing systems, such as Enterprise Resource Planning (ERP) or process control systems, can be complex and may require significant customization. Ensuring accurate data input is crucial, as MES systems rely on precise data to optimize operations, and incorrect information can lead to errors in production scheduling or inventory management. Additionally, employees must be trained to effectively use the MES system, which may require overcoming resistance to change and adapting to new workflows. Adequate support and continuous monitoring are also necessary to ensure the system operates effectively post-implementation.

    \nNow use the following context items to answer the user query:
    {context}
    \nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:"""
    
    return base_prompt