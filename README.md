# RAG Pipeline with Pinecone

A Retrieval-Augmented Generation (RAG) implementation using Pinecone as the vector database, with both a Streamlit interface and Jupyter notebook examples.

## Prerequisites

- Python 3.0 or higher
- Pinecone API key
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/parikshith078/rag-pipeline 
cd rag-pipeline
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
- Create a `.env` file in the root directory
- Add your Pinecone API key:
```
PINECONE_API_KEY=your_api_key_here
```

## Usage

### Streamlit Interface

Run the Streamlit application:
```bash
streamlit run streamlit_app/app.py
```

### Jupyter Notebooks

The project includes two Jupyter notebooks for different stages of the RAG pipeline:

1. **Data Preprocessing and Storage**
   - Preprocessing of source data
   - Embedding generation
   - Storage in Pinecone vector database

2. **RAG Pipeline Implementation**
   - Implementation of the retrieval-augmented generation pipeline
   - Query processing and response generation

To use the notebooks:
```bash
jupyter notebook
```

Navigate to the notebooks directory and open either notebook to begin.

## Project Structure
```
├── requirements.txt
├── .env
├── assets
├── rag_docs
├── streamlit_app/
│   └── app.py
└── notebooks/
    ├── rag_preprocessing_embedding.ipynb
    └── rag_workflow.ipynb
```

