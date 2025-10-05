# PDF and Image Retrieval with LangChain and CLIP

This notebook demonstrates how to build a multi-modal retrieval system using LangChain, FAISS, and the CLIP model to search for relevant information within a PDF document, including both text and images.

## Setup

### Install Libraries

This cell installs the necessary libraries for the project, including `langchain-community`, `sentence-transformers`, `faiss-cpu`, `pdf2image`, `Pillow`, `python-dotenv`, `pypdf`, and `pymupdf`.

### Upload .env and PDF

This cell handles the uploading of your `.env` file containing the `PERPLEXITY_API_KEY` and your PDF file. It then loads the environment variables and extracts the PDF filename for later use.

### Load PDF

This cell uses `PyPDFLoader` from `langchain` to load the uploaded PDF document into a list of `Document` objects. It also prints the number of pages loaded and a preview of the first page.

### Chunk text

These cells use `RecursiveCharacterTextSplitter` to split the loaded PDF documents into smaller text chunks. This is done to manage the size of the text fed into the embedding model and improve retrieval accuracy.

### Extract images from PDF

These cells utilize `fitz` (PyMuPDF) to open the PDF and extract images from each page. The extracted images are stored as PIL Image objects along with their corresponding page number.

### Create embeddings for text and images

These cells use the CLIP model from the `transformers` library to create embeddings for both the text chunks and the extracted images. Embeddings are numerical representations of the content, allowing for similarity comparisons. The embeddings are stored as NumPy arrays.

### FAISS

These cells use the FAISS library to create vector indices for the text and image embeddings. FAISS is an efficient library for similarity search and clustering of dense vectors. Separate indices are created for text and image embeddings.

### Create retrievers

These cells define custom callable classes for CLIP text and image embeddings to be compatible with LangChain's `FAISS` vectorstore. It then creates `FAISS` retrievers for both text and images using the created indices and in-memory docstores. Finally, a `MultiRetriever` class is defined to combine the text and image retrievers, allowing for searches across both modalities.

### Save embeddings + metadata to JSON

These cells iterate through the text chunks and image items, combining their content (or a preview), embedding, and type into a list of dictionaries. This data is then saved to a JSON file named `embeddings.json`.

### Retrieve relevant documents

This cell demonstrates how to use the `combined_retriever` to retrieve relevant documents (both text chunks and images) based on a text query. The results are printed to the console.

### Initialize ChatPerplexity and Conversational Chain

These cells install the `langchain-openai` library, set up the `PERPLEXITY_API_KEY` as an environment variable, initialize the `ChatPerplexity` language model, define a prompt template for the conversational chain, and create a `ConversationalRetrievalChain` using the language model, the combined retriever, and a memory buffer to maintain conversation history.

### Invoke the conversational chain

This cell invokes the conversational chain with a query "tell about this pdf" and prints the response. Please note that there might be an API error in this cell depending on your API key and usage limits.
