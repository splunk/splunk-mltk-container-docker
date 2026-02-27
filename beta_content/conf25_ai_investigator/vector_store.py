#---------------------------------------------------------------------------------
# Vector embedding and similarity search functionality
#---------------------------------------------------------------------------------

import os
import re
import glob
import math
import torch
import numpy as np
import faiss
import logfire
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

#----------------------------------------------------------------------------
def test_faiss_gpu():
    """Test if FAISS is using GPU"""
    try:
        res = faiss.StandardGpuResources()
        dimension = 64
        n_vectors = 1000
        cpu_index = faiss.IndexFlatL2(dimension)
        vectors = np.random.random((n_vectors, dimension)).astype(np.float32)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(vectors)
        print("✅ FAISS is successfully using GPU")
        print(f"Number of vectors in index: {gpu_index.ntotal}")
        return True
    except Exception as e:
        print("❌ FAISS GPU test failed")
        print(f"Error: {str(e)}")
        return False
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Initialize vector store
class VectorStore:
    def __init__(self, dimension: int = 512, embedding_model: str = "local"):
        self.use_gpu = torch.cuda.is_available()
        logfire.info("Initializing VectorStore",
                     use_gpu=self.use_gpu,
                     dimension=dimension,
                     embedding_model=embedding_model)
        
        if self.use_gpu:
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            self.res = faiss.StandardGpuResources()
        
        self.embedding_model = embedding_model
        
        if embedding_model == "local":
            logfire.info("Loading local sentence transformer model")
            # self.model = SentenceTransformer('sentence-transformers/LaBSE')
            # self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            # self.model = SentenceTransformer('BAAI/bge-small-en-v1.5') # Bad
            # self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            # self.model = SentenceTransformer('BAAI/bge-m3')
            # self.model = SentenceTransformer('intfloat/e5-large-v2')
            if self.use_gpu:
                self.model.to('cuda')
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            self.dimension = dimension
        
        self.indices = {
            "data_description": self._create_index(),
            "query_example": self._create_index(),
            "tool_description": self._create_index()
        }
        
        self.documents = {k: [] for k in self.indices}
        self.metadatas = {k: [] for k in self.indices}
        self.ids = {k: [] for k in self.indices}
        
        logfire.info("VectorStore initialized successfully")

    def _create_index(self):
        cpu_index = faiss.IndexFlatIP(self.dimension)
        if self.use_gpu:
            return faiss.index_cpu_to_gpu(self.res, 0, cpu_index)
        return cpu_index
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.embedding_model == "local":
            with torch.no_grad():
                batch_size = 32
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_tensor=True,
                        device='cuda' if self.use_gpu else 'cpu',
                        batch_size=batch_size,
                        show_progress_bar=False
                    )
                    if self.use_gpu:
                        batch_embeddings = batch_embeddings.cpu()
                    embeddings.append(batch_embeddings.numpy())
                return np.vstack(embeddings)
        else:
            # OpenAI embeddings if needed
            client = OpenAI()
            embeddings_batch = []
            for text in texts:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                embeddings_batch.append(response.data[0].embedding)
            return np.array(embeddings_batch, dtype=np.float32)
    
    def add_documents(self, documents: List[Dict], ids: List[str]):
        logfire.info("Adding documents to VectorStore", num_documents=len(documents))
        grouped_docs = {}
        grouped_ids = {}
        
        for doc, id in zip(documents, ids):
            entry_type = doc["metadata"]["entry_type"]
            if entry_type not in grouped_docs:
                grouped_docs[entry_type] = []
                grouped_ids[entry_type] = []
            grouped_docs[entry_type].append(doc)
            grouped_ids[entry_type].append(id)
        
        for entry_type, docs in grouped_docs.items():
            if entry_type not in self.indices:
                logfire.info("Skipping unknown entry type", entry_type=entry_type, level="warn")
                continue
                
            texts = [doc["page_content"] for doc in docs]
            logfire.debug("Getting embeddings",
                          entry_type=entry_type,
                          num_texts=len(texts))
            embeddings_array = self.get_embeddings(texts)
            embeddings_array = np.ascontiguousarray(embeddings_array.astype(np.float32))
            # NEW: Normalize document embeddings (L2 normalization)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-10
            embeddings_array = embeddings_array / norms

            # Debug check for datastore description embeddings differences.
            # import hashlib
            # for i, doc in enumerate(docs):
            #     checksum = hashlib.md5(embeddings_array[i].tobytes()).hexdigest()
            #     logfire.debug("Doc embedding checksum",
            #                   doc_id=grouped_ids[entry_type][i],
            #                   checksum=checksum,
            #                   page_content_preview=doc["page_content"][:100])
            
            self.indices[entry_type].add(embeddings_array)
            self.documents[entry_type].extend(texts)
            self.metadatas[entry_type].extend([doc["metadata"] for doc in docs])
            self.ids[entry_type].extend(grouped_ids[entry_type])
            logfire.info("Documents added successfully",
                         entry_type=entry_type,
                         num_docs=len(docs))

    def similarity_search_with_score(self, query: str, k: int = 4, filter: Dict = None) -> List[Tuple[Dict, float]]:
        if not filter or "entry_type" not in filter:
            logfire.info("Missing entry_type filter", level="warn")
            raise ValueError("entry_type filter is required")
            
        entry_type = filter["entry_type"]
        if entry_type not in self.indices:
            logfire.info("Invalid entry type for search", entry_type=entry_type, level="warn")
            return []
            
        if len(self.documents[entry_type]) == 0:
            logfire.info("No documents found for entry type", entry_type=entry_type, level="warn")
            return []
            
        logfire.debug("Getting query embedding", query=query)
        query_embedding = self.get_embeddings([query])
        # NEW: Normalize the query embedding (L2 normalization)
        norms = np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-10
        query_embedding = query_embedding / norms
        
        try:
            D, I = self.indices[entry_type].search(query_embedding, min(k, len(self.documents[entry_type])))
            logfire.debug("Search completed", num_results=len(I[0]))
            # NEW: Log the raw cosine similarity scores returned by FAISS for debugging.
            logfire.debug("FAISS Scores", scores=D[0].tolist())
        except AssertionError:
            logfire.error("Search failed")
            return []
        
        results = []
        for distance, doc_idx in zip(D[0], I[0]):
            if doc_idx < 0:
                continue
            doc = {
                "page_content": self.documents[entry_type][doc_idx],
                "metadata": self.metadatas[entry_type][doc_idx]
            }
            # Log match details including datastore name and match score
            logfire.debug("Match Found for DataStore",
                          doc_idx=doc_idx,
                          similarity=distance,
                          data_store=doc["metadata"].get("data_store", "N/A"),
                          data_store_name=doc["metadata"].get("data_store_name", "N/A"))
            
            filter_copy = filter.copy()
            del filter_copy["entry_type"]
            if filter_copy:
                matches_filter = True
                for key, value in filter_copy.items():
                    if doc["metadata"].get(key) != value:
                        matches_filter = False
                        break
                if not matches_filter:
                    continue
                    
            results.append((doc, float(distance)))
        
        logfire.info("Similarity search completed",
                     entry_type=entry_type,
                     num_results=len(results),
                     k=k)
        return results

    def delete(self, ids: List[str]):
        logfire.info("Deleting documents", num_ids=len(ids))
        for entry_type in self.indices:
            new_documents = []
            new_metadatas = []
            new_ids = []
            embeddings_to_keep = []
            
            for i, doc_id in enumerate(self.ids[entry_type]):
                if doc_id not in ids:
                    embeddings_to_keep.append(
                        self.get_embeddings([self.documents[entry_type][i]])[0]
                    )
                    new_documents.append(self.documents[entry_type][i])
                    new_metadatas.append(self.metadatas[entry_type][i])
                    new_ids.append(doc_id)
            
            self.indices[entry_type] = self._create_index()
            if embeddings_to_keep:
                embeddings_array = np.array(embeddings_to_keep, dtype=np.float32)
                self.indices[entry_type].add(embeddings_array)
            
            self.documents[entry_type] = new_documents
            self.metadatas[entry_type] = new_metadatas
            self.ids[entry_type] = new_ids
            logfire.info("Documents deleted successfully",
                         entry_type=entry_type,
                         remaining_docs=len(new_documents))

    def get(self) -> Dict:
        return {
            "documents": self.documents,
            "metadatas": self.metadatas,
            "ids": self.ids
        }
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def parse_query_example(content: str) -> List[Dict]:
    """Parse query examples from content and return list of documents to upsert"""
    documents = []
    
    # Filter out lines that start with '#'
    filtered_lines = [line for line in content.split('\n') if not line.strip().startswith('#')]
    content = '\n'.join(filtered_lines)
    
    # Split content into individual examples
    examples = re.split(r'(?i)(?=\s*User query:)', content)
    examples = [ex for ex in examples if re.match(r'(?i)\s*User Query:', ex)]
    
    for example in examples:
        # Reset for each example
        user_query = None
        spl_answer = None
        
        # Find the user query
        user_query_match = re.search(r'(?i)\s*User Query:\s*(.*?)(?=\s*SPL answer:|$)', example, re.DOTALL)
        if user_query_match:
            user_query = user_query_match.group(1).strip()
        
        # Find the SPL answer - look for everything after "SPL answer:" until the next example or end of text
        spl_match = re.search(r'(?i)\s*SPL answer:\s*(.*?)(?=(?:\s*User query:)|$)', example, re.DOTALL)
        if spl_match:
            spl_answer = spl_match.group(1).strip()
        
        if user_query and spl_answer:
            user_query_name = re.sub(r'[^\w]+', '_', user_query).lower().strip('_')
            combined_content = f"User query: {user_query}\nSPL answer: {spl_answer}"
            
            documents.append({
                "page_content": combined_content,
                "metadata": {
                    "entry_type": "query_example",
                    "user_query_name": user_query_name,
                    "id": user_query_name
                }
            })
    
    return documents
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def parse_datastore_description(content: str) -> Optional[Dict]:
    """Parse datastore description content and return document to upsert"""
    # Filter out lines that start with '#'
    filtered_lines = [line for line in content.split('\n') if not line.strip().startswith('#')]
    content = '\n'.join(filtered_lines)
    
    data_store_type = None
    data_store_name = None
    description_lines = []
    content_lines = []
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Handle field names that have values on the next line
        if line == 'Data_Store_Type:':
            if i + 1 < len(lines):
                data_store_type = lines[i + 1].strip()
                i += 2
                continue
        elif line == 'Data_Store_Name:':
            if i + 1 < len(lines):
                data_store_name = lines[i + 1].strip()
                i += 2
                continue
        elif line == 'Data_Store_Description:':
            i += 1  # Move to first line of description
            # Collect all indented lines as description
            while i < len(lines) and lines[i].startswith('  '):
                description_line = lines[i].strip()
                if description_line:  # Only add non-empty lines
                    description_lines.append(description_line)
                i += 1
                continue
        elif line == 'Data_Store_Content:':
            i += 1  # Skip the header line
            # Collect all remaining lines as content
            while i < len(lines):
                content_line = lines[i].strip()
                if content_line:  # Only add non-empty lines
                    content_lines.append(content_line)
                i += 1
        else:
            i += 1
    
    if data_store_type and data_store_name:
        # Combine Data_Store_Type, Data_Store_Name, Description, and Content into page_content.
        combined_text = f"Data_Store_Type: {data_store_type}\n"
        combined_text += f"Data_Store_Name: {data_store_name}\n"
        if description_lines:
            combined_text += "Data_Store_Description:\n" + '\n'.join(description_lines).strip() + "\n"
        if content_lines:
            combined_text += "Data_Store_Content:\n" + '\n'.join(content_lines).strip()
        return {
            "page_content": combined_text,
            "metadata": {
                "entry_type": "data_description",
                "data_store": data_store_type,
                "data_store_name": data_store_name,
                "description": '\n'.join(description_lines).strip() if description_lines else None,
                "id": f"{data_store_type}_{data_store_name}"
            }
        }
    return None
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
def load_vector_db_content():
    """Load vector DB content from files in VECTOR_DB_CONTENT directory"""
    base_dir = os.path.join(os.path.dirname(__file__), 'VECTOR_DB_CONTENT')
    documents_to_upsert = []
    
    # Load query samples
    samples_dir = os.path.join(base_dir, 'QUERY_SAMPLES')
    if os.path.exists(samples_dir):
        for filepath in glob.glob(os.path.join(samples_dir, '**/*.txt'), recursive=True):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count and log comment lines
                comment_lines = len([line for line in content.split('\n') if line.strip().startswith('#')])
                if comment_lines > 0:
                    logfire.info(f"Skipping {comment_lines} comment lines in file: {filepath}")
                    
                documents_to_upsert.extend(parse_query_example(content))
                logfire.info(f"Processed query samples file: {filepath}")
            except Exception as e:
                logfire.error(f"Error processing file {filepath}: {str(e)}")
    
    # Load datastore descriptions
    datastores_dir = os.path.join(base_dir, 'DATASTORES')
    if os.path.exists(datastores_dir):
        for filepath in glob.glob(os.path.join(datastores_dir, '*.txt')):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count and log comment lines
                comment_lines = len([line for line in content.split('\n') if line.strip().startswith('#')])
                if comment_lines > 0:
                    logfire.info(f"Skipping {comment_lines} comment lines in file: {filepath}")
                    
                if doc := parse_datastore_description(content):
                    documents_to_upsert.append(doc)
                logfire.info(f"Processed datastore file: {filepath}")
            except Exception as e:
                logfire.error(f"Error processing file {filepath}: {str(e)}")
    
    # Add all documents to vector store if any were found
    if documents_to_upsert:
        vector_store = VectorStore(embedding_model="local")
        vector_store.add_documents(
            documents_to_upsert,
            ids=[doc["metadata"]["id"] for doc in documents_to_upsert]
        )
        logfire.info(f"Added {len(documents_to_upsert)} documents to vector store")
        return vector_store
    else:
        logfire.warning("No documents found to load into vector store")
        return VectorStore(embedding_model="local")
#----------------------------------------------------------------------------
