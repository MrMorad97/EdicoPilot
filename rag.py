import logging
import hashlib
import re
from functools import lru_cache
from typing import Generator, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LangChain components
try:
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import ChatOllama
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import PromptTemplate
    from langchain_community.vectorstores.utils import filter_complex_metadata
    from langchain_core.documents import Document
    logger.info("LangChain components imported successfully")
except ImportError as e:
    logger.error(f"Failed to import LangChain components: {e}")
    raise

class StreamingChatPDF:
    def __init__(self):
        # Model configuration with balanced settings
        self.model = ChatOllama(
            model="mistral",
            base_url="http://127.0.0.1:11434",
            temperature=0.7,
            timeout=120,
            num_ctx=2048
        )
        
        # Optimized text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Efficient embeddings
        self.embedding = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir="embeddings_cache"
        )
        
        self.vector_store = None
        self.retriever = None
        self.has_documents = False
        self.response_cache = {}
        self.default_prompt = """
        Answer the question based on the following context. Be concise and accurate.
        
        Context: {context}
        Question: {question}
        
        Answer:
        """

    def _clean_text(self, text: str) -> str:
        """Clean text while preserving structure"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Page \d+', '', text)
        return text.strip()

    def _get_cache_key(self, question: str, context: str = "") -> str:
        """Generate consistent cache key"""
        return hashlib.md5(f"{question}_{context[:300]}".encode()).hexdigest()

    def _format_context(self, docs: list[Document], max_length: int = 2500) -> str:
        """Format context with length limits"""
        if not docs:
            return ""
            
        contexts = []
        total_length = 0
        
        for doc in docs:
            content = self._clean_text(doc.page_content)
            if total_length + len(content) <= max_length:
                contexts.append(content)
                total_length += len(content)
            else:
                remaining = max_length - total_length
                if remaining > 50:
                    contexts.append(content[:remaining] + "...")
                break
        
        return "\n\n".join(contexts)

    def ingest_pdfs(self, pdf_paths: list[str]) -> bool:
        """Load and process PDFs efficiently"""
        try:
            self.clear()
            
            if not pdf_paths:
                logger.info("Direct chat mode - no PDFs")
                return False

            docs = []
            for path in pdf_paths:
                try:
                    loaded_docs = PyPDFLoader(path).load()
                    for doc in loaded_docs:
                        doc.page_content = self._clean_text(doc.page_content)
                        if len(doc.page_content) > 30:  # Filter very short chunks
                            docs.append(doc)
                except Exception as e:
                    logger.warning(f"Skipped {path}: {str(e)}")
                    continue

            if not docs:
                logger.warning("No valid PDF content after filtering")
                return False

            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding,
                persist_directory="./chroma_db",
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            # Configure retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "score_threshold": 0.4,
                    "fetch_k": 20
                }
            )
            
            self.has_documents = True
            return True
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}", exc_info=True)
            return False

    def stream_response(
        self, 
        question: str, 
        prompt_template: str = None
    ) -> Generator[str, None, None]:
        """Stream response with proper prompt handling"""
        try:
            # Use provided template or default
            template = prompt_template if prompt_template else self.default_prompt
            prompt = PromptTemplate.from_template(template)
            
            if self.has_documents:
                # Retrieve relevant documents
                docs = self.retriever.get_relevant_documents(question)
                context = self._format_context(docs)
                
                # Check cache
                cache_key = self._get_cache_key(question, context)
                if cache_key in self.response_cache:
                    logger.info("Using cached response")
                    yield self.response_cache[cache_key]
                    return
                
                # Format final prompt
                formatted_prompt = prompt.format(
                    question=question,
                    context=context
                )
                
                # Stream response
                full_response = ""
                for chunk in self.model.stream(formatted_prompt):
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    full_response += content
                    yield content
                
                # Update cache
                if len(self.response_cache) > 50:
                    self.response_cache.pop(next(iter(self.response_cache)))
                self.response_cache[cache_key] = full_response
                
            else:
                # Direct chat
                for chunk in self.model.stream(question):
                    yield chunk.content if hasattr(chunk, 'content') else str(chunk)
                    
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"

    def clear(self):
        """Clean up resources"""
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
            except Exception as e:
                logger.warning(f"Error clearing vector store: {str(e)}")
        self.vector_store = None
        self.retriever = None
        self.has_documents = False
        self.response_cache.clear()

# Module initialization
if __name__ == "__main__":
    try:
        test = StreamingChatPDF()
        logger.info("Module self-test passed")
    except Exception as e:
        logger.error(f"Module self-test failed: {e}")