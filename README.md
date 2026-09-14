# PP Knowledge Assistant with Advanced RAG

A Streamlit-based Retrieval-Augmented Generation (RAG) assistant for searching internal company documents.

It combines semantic vector search, BM25 keyword search, document reranking, and OpenAI answer generation to provide grounded responses with source references.

## Features

- Local document ingestion
- Supports PDF, DOCX, TXT, MD
- Chunking and embeddings
- Local vector store using ChromaDB
- Hybrid retrieval: Vector + BM25
- Reranking using Cross-Encoder
- Conversational memory
- Source citations
- Streamlit UI
- Hallucination mitigation

## Project Structure

.
├── EnterpriseKnowledgeAssistantApp.py
├── requirements.txt
├── .env
├── data/
│   ├── raw_docs/              # Add source documents here
│   ├── chroma_db/             # Generated vector database
│   ├── processed_chunks.jsonl # Generated document chunks
│   └── user_memory.json       # Local saved user facts
└── src/
    ├── config.py
    ├── ingest.py
    ├── loaders.py
    ├── llm.py
    ├── memory.py
    ├── reranker.py
    ├── retriever.py
    └── utils.py

## Requirements
	Python 3.10+
	OpenAI API key
	Internet access for downloading Hugging Face models

## Installation

### 1. Create and activate a virtual environment
	```bash
		python -m venv .venv
	```
### 2. Activate environment

#### Windows
	```bash
	.venv\Scripts\activate
	```

#### Mac/Linux
	```bash
	source .venv/bin/activate
	```
### 3. Install dependencies
	```bash
	pip install -r requirements.txt
	```		
	
### 4. Configure environment
	Rename a `.env.example` to `.env` file in the project root
	# OpenAPI Details
	OPENAI_API_KEY=your_openai_api_key
	OPENAI_MODEL=gpt-4o-mini
	
	# Gemini API Details
	GEMINI_API_KEY=your_gemini_api_key
	GEMINI_MODEL=gemini-2.5-flash
	
	# You can choose "openai" or "gemini" as option in LLM_PROVIDER
	LLM_PROVIDER=openai	

### 5. Add documents
	Place company documents in:
	data/raw_docs/

	Supported formats:
	.pdf
	.docx
	.txt
	.md

### 6. Run the application
	```bash
	streamlit run EnterpriseKnowledgeAssistantApp.py
	```
	Then open the local URL shown by Streamlit, usually: http://localhost:8501

## Build the Knowledge Base
	From the project root, run:
		python -m src.ingest --rebuild
	This loads the documents, creates chunks, generates embeddings, and stores the index in data/chroma_db/

## Configuration
	Main settings are available in:	
	src/config.py
	Important options include:
		1.Chunk size and overlap
		2.Embedding model
		3.Reranker model
		4.Number of retrieved documents
		5.ChromaDB storage location
	Default models:
		sentence-transformers/all-MiniLM-L6-v2
		cross-encoder/ms-marco-MiniLM-L-6-v2

## Memory
	Conversational memory can be enabled or disabled from the sidebar.	
