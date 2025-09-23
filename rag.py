# rag.py - RAG (Retrieval Augmented Generation) module for LLM Interactive Client
# This module provides document processing, vector database management, and retrieval functionality

import os
import glob
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

import config
# Note: Conversation chain imports removed as they're not used in the current implementation

class RAGManager:
    """
    Manages RAG functionality including document loading, vector database operations,
    and retrieval chain management.
    """

    def __init__(self, openai_api_key=None, db_name=None, embedding_provider=None, embedding_model=None, embedding_base_url=None):
        """
        Initialize RAG Manager.

        Args:
            openai_api_key (str): OpenAI API key for embeddings
            db_name (str): Name of the vector database directory (defaults to config value)
            embedding_provider (str): Embedding provider ("OpenAI" or "Custom")
            embedding_model (str): Embedding model name
            embedding_base_url (str): Custom embedding base URL
        """
        self.db_name = db_name or config.DEFAULT_VECTOR_DB_NAME
        self.vectorstore = None
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.embedding_provider = embedding_provider or config.DEFAULT_EMBEDDING_PROVIDER
        self.embedding_model = embedding_model or config.DEFAULT_EMBEDDING_MODEL
        self.embedding_base_url = embedding_base_url

        # Initialize embeddings if API key is provided
        if openai_api_key:
            self.set_embeddings(openai_api_key, self.embedding_provider, self.embedding_model, self.embedding_base_url)

    def set_api_key(self, api_key):
        """Set or update the OpenAI API key for embeddings (legacy method)."""
        return self.set_embeddings(api_key, self.embedding_provider, self.embedding_model, self.embedding_base_url)

    def set_embeddings(self, api_key, provider=None, model=None, base_url=None):
        """
        Set or update embedding configuration.

        Args:
            api_key (str): API key for embeddings (can be empty for local services)
            provider (str): Embedding provider ("OpenAI" or "Custom")
            model (str): Embedding model name
            base_url (str): Custom embedding base URL
        """
        provider = provider or self.embedding_provider
        model = model or self.embedding_model
        base_url = base_url or self.embedding_base_url

        # API key is required for all providers
        if not api_key or not api_key.strip():
            return False

        try:
            if provider == "OpenAI":
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=api_key,
                    model=model
                )
            elif provider == "Custom":
                # Use OpenAI-compatible embedding endpoint with custom base URL
                print(f"Setting custom embeddings with base_url: {base_url}, model: {model}")
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=api_key,
                    model=model,
                    base_url=base_url,
                    check_embedding_ctx_length=False  # Disable context length checking for custom endpoints
                )
            else:
                print(f"Unknown provider: {provider}")
                return False

            # Update instance variables
            self.embedding_provider = provider
            self.embedding_model = model
            self.embedding_base_url = base_url

            print(f"Successfully set embeddings: provider={provider}, model={model}")
            return True
        except Exception as e:
            print(f"Error setting embeddings: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_documents_from_folder(self, knowledge_base_path, file_pattern=None):
        """
        Load documents from a knowledge base folder structure.

        Args:
            knowledge_base_path (str): Path to the knowledge base folder
            file_pattern (str): File pattern to match (defaults to config value)

        Returns:
            tuple: (success, message, document_count)
        """
        try:
            if not knowledge_base_path:
                return False, f"Knowledge base path not provided", 0

            # Use config default if no pattern specified
            if not file_pattern:
                file_pattern = config.RAG_FILE_PATTERNS[0]  # Use first pattern as default

            if not os.path.exists(knowledge_base_path):
                return False, f"Knowledge base path '{knowledge_base_path}' does not exist", 0

            # Get all subfolders or use the main folder if no subfolders
            folders = glob.glob(os.path.join(knowledge_base_path, "*"))
            folders = [f for f in folders if os.path.isdir(f)]

            if not folders:
                # No subfolders, use the main folder
                folders = [knowledge_base_path]

            def add_metadata(doc, doc_type):
                doc.metadata["doc_type"] = doc_type
                return doc

            # Text loader configuration for different encodings
            text_loader_kwargs = {'encoding': 'utf-8'}

            documents = []
            for folder in folders:
                doc_type = os.path.basename(folder)
                try:
                    loader = DirectoryLoader(
                        folder,
                        glob=file_pattern,
                        loader_cls=TextLoader,
                        loader_kwargs=text_loader_kwargs
                    )
                    folder_docs = loader.load()
                    documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])
                except Exception as e:
                    print(f"Warning: Could not load documents from {folder}: {e}")

            if not documents:
                return False, "No documents found in the specified path", 0

            # Split documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=config.DEFAULT_CHUNK_SIZE, chunk_overlap=config.DEFAULT_CHUNK_OVERLAP)
            chunks = text_splitter.split_documents(documents)

            self.documents = documents
            self.chunks = chunks

            doc_types = set(doc.metadata.get('doc_type', 'unknown') for doc in documents)
            return True, f"Successfully loaded {len(documents)} documents ({len(chunks)} chunks) from types: {', '.join(doc_types)}", len(documents)

        except Exception as e:
            return False, f"Error loading documents: {str(e)}", 0

    def create_vector_database(self):
        """
        Create vector database from loaded documents.

        Returns:
            tuple: (success, message)
        """
        try:
            if not self.embeddings:
                return False, "OpenAI API key not set. Please provide a valid API key."

            if not self.chunks:
                return False, "No documents loaded. Please load documents first."

            # Delete existing database if it exists
            if os.path.exists(self.db_name):
                try:
                    existing_vectorstore = Chroma(persist_directory=self.db_name, embedding_function=self.embeddings)
                    existing_vectorstore.delete_collection()
                except Exception as e:
                    print(f"Warning: Could not delete existing collection: {e}")

            # Create new vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=self.chunks,
                embedding=self.embeddings,
                persist_directory=self.db_name
            )

            count = self.vectorstore._collection.count()
            return True, f"Vector database created successfully with {count} document chunks."

        except Exception as e:
            return False, f"Error creating vector database: {str(e)}"

    def load_existing_vector_database(self):
        """
        Load existing vector database.

        Returns:
            tuple: (success, message)
        """
        try:
            if not self.embeddings:
                return False, "OpenAI API key not set. Please provide a valid API key."

            if not os.path.exists(self.db_name):
                return False, f"Vector database '{self.db_name}' does not exist."

            self.vectorstore = Chroma(persist_directory=self.db_name, embedding_function=self.embeddings)
            count = self.vectorstore._collection.count()

            if count == 0:
                return False, "Vector database exists but is empty."

            return True, f"Successfully loaded existing vector database with {count} documents."

        except Exception as e:
            return False, f"Error loading vector database: {str(e)}"


    def get_retrieval_context(self, question, k=None):
        """
        Get retrieved context for a question without generating a response.

        Args:
            question (str): User question
            k (int): Number of documents to retrieve (defaults to config value)

        Returns:
            list: Retrieved documents with similarity scores
        """
        try:
            if not self.vectorstore:
                return []

            # Use config default if k not specified
            if k is None:
                k = config.DEFAULT_RETRIEVAL_K

            # Use similarity search with scores to get relevance information
            retrieved_docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)

            # Add score information to document metadata
            for doc, score in retrieved_docs_with_scores:
                doc.metadata['similarity_score'] = score
                doc.metadata['similarity_percentage'] = max(0, (1 - score) * 100)  # Convert distance to similarity %

            # Return just the documents (now with score metadata)
            return [doc for doc, _ in retrieved_docs_with_scores]

        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []

    def create_vector_visualization(self, dimensions=2, sample_size=None):
        """
        Create 2D or 3D visualization of the vector database.

        Args:
            dimensions (int): 2 or 3 for 2D or 3D visualization
            sample_size (int): Maximum number of vectors to visualize (defaults to config value)

        Returns:
            plotly.graph_objects.Figure or None
        """
        try:
            if not self.vectorstore:
                print("Debug: No vectorstore available")
                return None

            # Use config default if sample_size not specified
            if sample_size is None:
                sample_size = config.MAX_VISUALIZATION_VECTORS

            collection = self.vectorstore._collection
            count = collection.count()
            print(f"Debug: Collection count: {count}")

            if count == 0:
                print("Debug: Collection is empty")
                return None

            # Sample vectors if too many
            limit = min(sample_size, count)
            print(f"Debug: Fetching {limit} vectors")
            result = collection.get(limit=limit, include=['embeddings', 'documents', 'metadatas'])

            print(f"Debug: Result keys: {list(result.keys()) if result else 'None'}")

            # Safer boolean check - avoid numpy array boolean ambiguity
            if result is None:
                print("Debug: No result from collection.get()")
                return None

            embeddings_data = result.get('embeddings')
            if embeddings_data is None:
                print("Debug: No embeddings key in result")
                return None

            if len(embeddings_data) == 0:
                print("Debug: Empty embeddings list")
                return None

            print(f"Debug: Number of embeddings: {len(embeddings_data)}")
            print(f"Debug: Type of embeddings_data: {type(embeddings_data)}")

            try:
                print("Debug: Creating numpy array from embeddings...")
                vectors = np.array(embeddings_data)
                print(f"Debug: Vector array shape: {vectors.shape}")
                print(f"Debug: Vector array dtype: {vectors.dtype}")

                print("Debug: Getting documents...")
                documents = result.get('documents', [])
                if documents is None:
                    documents = []
                print(f"Debug: Documents type: {type(documents)}, count: {len(documents)}")

                print("Debug: Getting metadatas...")
                metadatas = result.get('metadatas', [])
                if metadatas is None:
                    metadatas = []
                print(f"Debug: Metadatas type: {type(metadatas)}, count: {len(metadatas)}")

            except Exception as e:
                print(f"Error processing vector data: {e}")
                import traceback
                traceback.print_exc()
                return None

            # Ensure we have consistent data lengths using basic comparisons
            num_vectors = vectors.shape[0] if vectors.size > 0 else 0
            num_documents = len(documents)
            num_metadatas = len(metadatas)

            print(f"Debug: Lengths - vectors: {num_vectors}, documents: {num_documents}, metadatas: {num_metadatas}")

            if num_vectors != num_documents or num_vectors != num_metadatas:
                print("Warning: Inconsistent data lengths, truncating to shortest")
                min_len = min(num_vectors, num_documents, num_metadatas)
                vectors = vectors[:min_len]
                documents = documents[:min_len]
                metadatas = metadatas[:min_len]
                print(f"Debug: After truncation - vectors: {vectors.shape[0]}, documents: {len(documents)}, metadatas: {len(metadatas)}")

            # Get document types and assign colors
            doc_types = [metadata.get('doc_type', 'unknown') if metadata else 'unknown' for metadata in metadatas]
            unique_types = list(set(doc_types))
            color_map = {
                doc_type: ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray'][i % 8]
                for i, doc_type in enumerate(unique_types)
            }
            colors = [color_map[doc_type] for doc_type in doc_types]

            # Reduce dimensionality using t-SNE
            # Ensure we have enough vectors for t-SNE
            final_vector_count = vectors.shape[0] if vectors.size > 0 else 0
            print(f"Debug: Final vector count for t-SNE: {final_vector_count}")

            if final_vector_count < 2:
                print("Debug: Not enough vectors for t-SNE (need at least 2)")
                return None

            try:
                # Set appropriate perplexity (must be less than n_samples)
                perplexity = min(30, max(5, final_vector_count - 1))
                print(f"Debug: Using perplexity: {perplexity}")
                tsne = TSNE(n_components=dimensions, random_state=42, perplexity=perplexity)
                print(f"Debug: Starting t-SNE with {dimensions}D")
                reduced_vectors = tsne.fit_transform(vectors)
                print(f"Debug: t-SNE completed, reduced shape: {reduced_vectors.shape}")
            except Exception as e:
                print(f"Error in t-SNE computation: {e}")
                return None

            # Create hover text
            hover_text = []
            for doc_type, doc in zip(doc_types, documents):
                try:
                    # Safely truncate document text
                    doc_text = str(doc)[:100] + "..." if len(str(doc)) > 100 else str(doc)
                    hover_text.append(f"Type: {doc_type}<br>Text: {doc_text}")
                except Exception as e:
                    hover_text.append(f"Type: {doc_type}<br>Text: [Error displaying text]")

            if dimensions == 2:
                # Create 2D scatter plot
                fig = go.Figure(data=[go.Scatter(
                    x=reduced_vectors[:, 0],
                    y=reduced_vectors[:, 1],
                    mode='markers',
                    marker=dict(size=8, color=colors, opacity=0.8),
                    text=hover_text,
                    hoverinfo='text'
                )])

                fig.update_layout(
                    title=f'2D Vector Database Visualization ({len(vectors)} documents)',
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2',
                    width=800,
                    height=600,
                    margin=dict(r=20, b=10, l=10, t=40)
                )

            elif dimensions == 3:
                # Create 3D scatter plot
                fig = go.Figure(data=[go.Scatter3d(
                    x=reduced_vectors[:, 0],
                    y=reduced_vectors[:, 1],
                    z=reduced_vectors[:, 2],
                    mode='markers',
                    marker=dict(size=5, color=colors, opacity=0.8),
                    text=hover_text,
                    hoverinfo='text'
                )])

                fig.update_layout(
                    title=f'3D Vector Database Visualization ({len(vectors)} documents)',
                    scene=dict(
                        xaxis_title='Dimension 1',
                        yaxis_title='Dimension 2',
                        zaxis_title='Dimension 3'
                    ),
                    width=900,
                    height=700,
                    margin=dict(r=20, b=10, l=10, t=40)
                )

            return fig

        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None

    def get_database_stats(self):
        """
        Get statistics about the vector database.

        Returns:
            dict: Database statistics
        """
        try:
            if not self.vectorstore:
                return {"status": "No vector database loaded"}

            collection = self.vectorstore._collection
            count = collection.count()

            if count == 0:
                return {"status": "Vector database is empty"}

            # Get sample embedding to check dimensions
            try:
                sample = collection.get(limit=1, include=["embeddings"])
                embeddings = sample.get("embeddings") if sample else None
                if embeddings is not None and len(embeddings) > 0:
                    first_embedding = embeddings[0]
                    if first_embedding is not None:
                        dimensions = len(first_embedding)
                    else:
                        dimensions = 0
                else:
                    dimensions = 0
            except Exception as e:
                print(f"Error getting sample embedding: {e}")
                dimensions = 0

            # Get document types if available
            try:
                all_data = collection.get(include=["metadatas"])
                doc_types = {}
                if all_data and all_data.get("metadatas"):
                    for metadata in all_data["metadatas"]:
                        if metadata:  # Check if metadata is not None
                            doc_type = metadata.get("doc_type", "unknown")
                            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            except Exception as e:
                print(f"Error getting document types: {e}")
                doc_types = {}

            return {
                "status": "Active",
                "document_count": count,
                "embedding_dimensions": dimensions,
                "document_types": doc_types
            }

        except Exception as e:
            return {"status": f"Error: {str(e)}"}


