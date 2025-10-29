# Chapter 5: MCP and RAG (Retrieval-Augmented Generation)

## 5.1 Understanding RAG and MCP {#understanding-rag-and-mcp}

Retrieval-Augmented Generation (RAG) enhances LLM responses by retrieving relevant information from external knowledge bases. MCP supercharges RAG by providing a standardized way to access and integrate diverse data sources.

### 5.1.1 Traditional RAG Architecture {#traditional-rag-architecture}

```
User Query
    ↓
Embedding Model
    ↓
Vector Search
    ↓
Retrieved Documents
    ↓
Combine with Query → LLM → Response
```

### 5.1.2 MCP-Enhanced RAG Architecture {#mcp-enhanced-rag-architecture}

```
User Query
    ↓
┌─────────────────────────────────┐
│      MCP-Enabled RAG Pipeline  │
│                                 │
│ ┌──────────────────────────┐   │
│ │    Query Analysis (LLM)  │   │
│ └──────────┬───────────────┘   │
│            │                   │
│ ┌──────────▼───────────────┐   │
│ │   MCP Resource Discovery │   │
│ │   • Vector DB Server     │   │
│ │   • Document Server      │   │
│ │   • Database Server      │   │
│ │   • Web Search Server    │   │
│ └──────────┬───────────────┘   │
│            │                   │
│ ┌──────────▼───────────────┐   │
│ │     Parallel Retrieval   │   │
│ │  (Multiple MCP Servers)  │   │
│ └──────────┬───────────────┘   │
│            │                   │
│ ┌──────────▼───────────────┐   │
│ │     Context Ranking      │   │
│ └──────────┬───────────────┘   │
│            │                   │
│ ┌──────────▼───────────────┐   │
│ │     LLM Generation       │   │
│ └──────────┬───────────────┘   │
└─────────────┼───────────────────┘
              ↓
          Response
```

## 5.2 Building an MCP-Based RAG System {#building-an-mcp-based-rag-system}

### 5.2.1 Vector Database MCP Server {#vector-database-mcp-server}

```python
# vector_db_mcp_server.py
from mcp.server import Server
from mcp.types import Tool, Resource, ToolResult, TextContent
import chromadb
from sentence_transformers import SentenceTransformer
import json

class VectorDatabaseMCPServer(Server):
    """
    MCP Server for vector database operations
    Enables semantic search for RAG
    """
    
    def __init__(self, collection_name: str = "documents"):
        super().__init__("vector-db")
        
        # Initialize ChromaDB
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def list_tools(self) -> list[Tool]:
        return [
            Tool(
                name="semantic_search",
                description="Search documents using semantic similarity",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "n_results": {
                            "type": "number",
                            "description": "Number of results to return",
                            "default": 5
                        },
                        "filter": {
                            "type": "object",
                            "description": "Metadata filters"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="add_documents",
                description="Add documents to the vector database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "text": {"type": "string"},
                                    "metadata": {"type": "object"}
                                }
                            }
                        }
                    },
                    "required": ["documents"]
                }
            ),
            Tool(
                name="get_document",
                description="Retrieve specific document by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"}
                    },
                    "required": ["document_id"]
                }
            ),
            Tool(
                name="hybrid_search",
                description="Search using both semantic and keyword matching",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "n_results": {"type": "number", "default": 5}
                    },
                    "required": ["query"]
                }
            )
        ]
    
    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        try:
            if name == "semantic_search":
                return await self._semantic_search(
                    arguments["query"],
                    arguments.get("n_results", 5),
                    arguments.get("filter")
                )
            
            elif name == "add_documents":
                return await self._add_documents(arguments["documents"])
            
            elif name == "get_document":
                return await self._get_document(arguments["document_id"])
            
            elif name == "hybrid_search":
                return await self._hybrid_search(
                    arguments["query"],
                    arguments.get("keywords", []),
                    arguments.get("n_results", 5)
                )
        
        except Exception as e:
            return ToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )
    
    async def _semantic_search(self, query: str, n_results: int, filter_dict: dict = None):
        """Perform semantic search"""
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "query": query,
                    "results": formatted_results,
                    "count": len(formatted_results)
                }, indent=2)
            )]
        )
    
    async def _add_documents(self, documents: list):
        """Add documents to vector database"""
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts).tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=f"Successfully added {len(documents)} documents"
            )]
        )
    
    async def _get_document(self, document_id: str):
        """Retrieve specific document"""
        result = self.collection.get(ids=[document_id])
        
        if not result['ids']:
            return ToolResult(
                content=[TextContent(type="text", text="Document not found")],
                isError=True
            )
        
        document = {
            "id": result['ids'][0],
            "text": result['documents'][0],
            "metadata": result['metadatas'][0]
        }
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(document, indent=2)
            )]
        )
    
    async def _hybrid_search(self, query: str, keywords: list, n_results: int):
        """Combine semantic and keyword search"""
        # Semantic search
        semantic_results = await self._semantic_search(query, n_results * 2)
        semantic_data = json.loads(semantic_results.content[0].text)
        
        # Filter by keywords if provided
        if keywords:
            filtered_results = []
            for result in semantic_data["results"]:
                text_lower = result["text"].lower()
                if any(keyword.lower() in text_lower for keyword in keywords):
                    filtered_results.append(result)
        else:
            filtered_results = semantic_data["results"]
        
        # Return top n_results
        filtered_results = filtered_results[:n_results]
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "query": query,
                    "keywords": keywords,
                    "results": filtered_results,
                    "count": len(filtered_results)
                }, indent=2)
            )]
        )
    
    async def list_resources(self) -> list[Resource]:
        """Expose collection stats as resource"""
        return [
            Resource(
                uri="vectordb://stats",
                name="Vector Database Statistics",
                description="Current collection statistics",
                mimeType="application/json"
            )
        ]
    
    async def read_resource(self, uri: str):
        """Provide collection statistics"""
        if uri == "vectordb://stats":
            stats = {
                "collection_name": self.collection.name,
                "document_count": self.collection.count(),
                "embedding_dimension": 384  # for all-MiniLM-L6-v2
            }
            
            return {
                "contents": [
                    TextContent(
                        type="text",
                        text=json.dumps(stats, indent=2)
                    )
                ]
            }


# Run server
if __name__ == "__main__":
    server = VectorDatabaseMCPServer()
    server.run()
```

### 5.2.2 Document Processing MCP Server {#document-processing-mcp-server}

```python
# document_processor_mcp_server.py
from mcp.server import Server
from mcp.types import Tool, ToolResult, TextContent
import PyPDF2
import docx
from bs4 import BeautifulSoup
import markdown

class DocumentProcessorMCPServer(Server):
    """
    MCP Server for document processing
    Converts various formats to text for RAG
    """
    
    def __init__(self):
        super().__init__("document-processor")
    
    async def list_tools(self) -> list[Tool]:
        return [
            Tool(
                name="extract_text_from_pdf",
                description="Extract text content from PDF files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "pages": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Specific pages to extract (optional)"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="extract_text_from_docx",
                description="Extract text from Word documents",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"}
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="extract_text_from_html",
                description="Extract clean text from HTML",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "html_content": {"type": "string"},
                        "preserve_links": {
                            "type": "boolean",
                            "default": False
                        }
                    },
                    "required": ["html_content"]
                }
            ),
            Tool(
                name="chunk_text",
                description="Split text into chunks for RAG",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "chunk_size": {
                            "type": "number",
                            "default": 500,
                            "description": "Characters per chunk"
                        },
                        "overlap": {
                            "type": "number",
                            "default": 50,
                            "description": "Overlap between chunks"
                        },
                        "preserve_sentences": {
                            "type": "boolean",
                            "default": True
                        }
                    },
                    "required": ["text"]
                }
            )
        ]
    
    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        try:
            if name == "extract_text_from_pdf":
                return await self._extract_pdf(
                    arguments["file_path"],
                    arguments.get("pages")
                )
            
            elif name == "extract_text_from_docx":
                return await self._extract_docx(arguments["file_path"])
            
            elif name == "extract_text_from_html":
                return await self._extract_html(
                    arguments["html_content"],
                    arguments.get("preserve_links", False)
                )
            
            elif name == "chunk_text":
                return await self._chunk_text(
                    arguments["text"],
                    arguments.get("chunk_size", 500),
                    arguments.get("overlap", 50),
                    arguments.get("preserve_sentences", True)
                )
        
        except Exception as e:
            return ToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )
    
    async def _extract_pdf(self, file_path: str, pages: list = None):
        """Extract text from PDF"""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_parts = []
            
            if pages:
                pages_to_process = pages
            else:
                pages_to_process = range(len(pdf_reader.pages))
            
            for page_num in pages_to_process:
                page = pdf_reader.pages[page_num]
                text_parts.append(f"--- Page {page_num + 1} ---\\n")
                text_parts.append(page.extract_text())
            
            full_text = "\\n".join(text_parts)
            
            return ToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "file": file_path,
                        "pages_processed": len(pages_to_process),
                        "text": full_text,
                        "character_count": len(full_text)
                    }, indent=2)
                )]
            )
    
    async def _extract_docx(self, file_path: str):
        """Extract text from Word document"""
        doc = docx.Document(file_path)
        text_parts = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        full_text = "\\n\\n".join(text_parts)
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "file": file_path,
                    "paragraphs": len(text_parts),
                    "text": full_text,
                    "character_count": len(full_text)
                }, indent=2)
            )]
        )
    
    async def _extract_html(self, html_content: str, preserve_links: bool):
        """Extract clean text from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        if preserve_links:
            # Keep link text with URLs
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href:
                    link.string = f"{link.get_text()} ({href})"
        
        # Get text
        text = soup.get_text(separator='\\n')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        text = '\\n'.join(line for line in lines if line)
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "text": text,
                    "character_count": len(text)
                }, indent=2)
            )]
        )
    
    async def _chunk_text(self, text: str, chunk_size: int, overlap: int, preserve_sentences: bool):
        """Split text into overlapping chunks"""
        if preserve_sentences:
            # Simple sentence splitting
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if current_length + sentence_length > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(' '.join(current_chunk))
                    
                    # Start new chunk with overlap
                    overlap_sentences = []
                    overlap_length = 0
                    
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add final chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        else:
            # Simple character-based chunking
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - overlap
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "original_length": len(text),
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "num_chunks": len(chunks),
                    "chunks": chunks
                }, indent=2)
            )]
        )


if __name__ == "__main__":
    server = DocumentProcessorMCPServer()
    server.run()
```

### 5.2.3 Complete RAG Pipeline with MCP {#complete-rag-pipeline-with-mcp}

```python
# mcp_rag_pipeline.py
import anthropic
from mcp import Client as MCPClient
import asyncio
import json

class MCPRAGPipeline:
    """
    Complete RAG pipeline using MCP servers
    """
    
    def __init__(self, anthropic_api_key: str):
        self.llm = anthropic.Anthropic(api_key=anthropic_api_key)
        self.vector_db_client = None
        self.doc_processor_client = None
    
    async def initialize(self):
        """Initialize MCP connections"""
        # Connect to vector database server
        self.vector_db_client = MCPClient({
            "command": "python",
            "args": ["vector_db_mcp_server.py"]
        })
        await self.vector_db_client.connect()
        
        # Connect to document processor server
        self.doc_processor_client = MCPClient({
            "command": "python",
            "args": ["document_processor_mcp_server.py"]
        })
        await self.doc_processor_client.connect()
        
        print("MCP RAG Pipeline initialized")
    
    async def ingest_documents(self, file_paths: list):
        """
        Ingest documents into RAG system
        1. Extract text from documents
        2. Chunk text
        3. Add to vector database
        """
        all_chunks = []
        
        for file_path in file_paths:
            print(f"Processing: {file_path}")
            
            # Extract text based on file type
            if file_path.endswith('.pdf'):
                result = await self.doc_processor_client.call_tool(
                    "extract_text_from_pdf",
                    {"file_path": file_path}
                )
            elif file_path.endswith('.docx'):
                result = await self.doc_processor_client.call_tool(
                    "extract_text_from_docx",
                    {"file_path": file_path}
                )
            else:
                print(f"Unsupported file type: {file_path}")
                continue
            
            # Parse extraction result
            extraction_data = json.loads(result.content[0].text)
            text = extraction_data["text"]
            
            # Chunk the text
            chunk_result = await self.doc_processor_client.call_tool(
                "chunk_text",
                {
                    "text": text,
                    "chunk_size": 500,
                    "overlap": 50,
                    "preserve_sentences": True
                }
            )
            
            chunk_data = json.loads(chunk_result.content[0].text)
            
            # Prepare chunks for vector DB
            for i, chunk_text in enumerate(chunk_data["chunks"]):
                all_chunks.append({
                    "id": f"{file_path}:chunk{i}",
                    "text": chunk_text,
                    "metadata": {
                        "source": file_path,
                        "chunk_index": i,
                        "total_chunks": chunk_data["num_chunks"]
                    }
                })
        
        # Add all chunks to vector database
        if all_chunks:
            await self.vector_db_client.call_tool(
                "add_documents",
                {"documents": all_chunks}
            )
        
        print(f"Ingested {len(all_chunks)} chunks from {len(file_paths)} documents")
        
        return len(all_chunks)
    
    async def query(self, question: str, n_results: int = 5) -> str:
        """
        Query RAG system
        1. Search vector database
        2. Retrieve relevant chunks
        3. Generate response with LLM
        """
        # Semantic search
        search_result = await self.vector_db_client.call_tool(
            "semantic_search",
            {
                "query": question,
                "n_results": n_results
            }
        )
        
        # Parse search results
        search_data = json.loads(search_result.content[0].text)
        retrieved_docs = search_data["results"]
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(
                f"[Document {i+1}] (Source: {doc['metadata']['source']})\\n"
                f"{doc['text']}\\n"
            )
        
        context = "\\n---\\n".join(context_parts)
        
        # Generate response with LLM
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Answer the following question based on the provided context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
                }
            ]
        )
        
        # Extract text response
        answer = response.content[0].text
        
        # Add sources
        sources = list(set(doc['metadata']['source'] for doc in retrieved_docs))
        answer += f"\\n\\nSources:\\n" + "\\n".join(f"- {source}" for source in sources)
        
        return answer
    
    async def hybrid_query(self, question: str, keywords: list = None) -> str:
        """
        Advanced query with hybrid search
        """
        # Hybrid search (semantic + keyword)
        search_result = await self.vector_db_client.call_tool(
            "hybrid_search",
            {
                "query": question,
                "keywords": keywords or [],
                "n_results": 5
            }
        )
        
        search_data = json.loads(search_result.content[0].text)
        retrieved_docs = search_data["results"]
        
        if not retrieved_docs:
            return "No relevant documents found."
        
        # Build context
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(
                f"[Document {i+1}]\\n{doc['text']}\\n"
                f"(Source: {doc['metadata']['source']}, "
                f"Relevance: {1 - doc['distance']:.2f})"
            )
        
        context = "\\n---\\n".join(context_parts)
        
        # Generate response
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Answer the question based on the context.
Cite specific documents when making claims.

Context:
{context}

Question: {question}

Provide a detailed answer with citations."""
                }
            ]
        )
        
        return response.content[0].text
    
    async def conversational_rag(self, conversation_history: list) -> str:
        """
        RAG with conversation history
        """
        # Get the latest user message
        latest_query = conversation_history[-1]["content"]
        
        # Search with conversation context
        # In production, would use conversation history for better retrieval
        search_result = await self.vector_db_client.call_tool(
            "semantic_search",
            {
                "query": latest_query,
                "n_results": 3
            }
        )
        
        search_data = json.loads(search_result.content[0].text)
        retrieved_docs = search_data["results"]
        
        # Build context
        context = "\\n---\\n".join(doc["text"] for doc in retrieved_docs)
        
        # Add context to conversation
        messages = [
            {
                "role": "system",
                "content": f"Use this context to answer questions:\\n\\n{context}"
            }
        ] + conversation_history
        
        # Generate response
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=messages
        )
        
        return response.content[0].text


# Demo usage
async def demo_rag_pipeline():
    """Demonstrate complete RAG pipeline"""
    pipeline = MCPRAGPipeline(anthropic_api_key="your-key")
    await pipeline.initialize()
    
    print("=== MCP RAG Pipeline Demo ===\\n")
    
    # 1. Ingest documents
    print("Step 1: Ingesting documents...")
    documents = [
        "/path/to/document1.pdf",
        "/path/to/document2.docx",
    ]
    chunk_count = await pipeline.ingest_documents(documents)
    print(f"Ingested {chunk_count} chunks\\n")
    
    # 2. Simple query
    print("Step 2: Simple query...")
    question = "What are the key findings?"
    answer = await pipeline.query(question)
    print(f"Q: {question}")
    print(f"A: {answer}\\n")
    
    # 3. Hybrid query
    print("Step 3: Hybrid query with keywords...")
    answer = await pipeline.hybrid_query(
        "Explain the methodology",
        keywords=["research", "experiment"]
    )
    print(f"A: {answer}\\n")
    
    # 4. Conversational RAG
    print("Step 4: Conversational RAG...")
    conversation = [
        {"role": "user", "content": "What is the main topic?"},
    ]
    answer = await pipeline.conversational_rag(conversation)
    print(f"A: {answer}\\n")
    
    conversation.append({"role": "assistant", "content": answer})
    conversation.append({"role": "user", "content": "Tell me more about that"})
    
    answer = await pipeline.conversational_rag(conversation)
    print(f"A: {answer}")


if __name__ == "__main__":
    asyncio.run(demo_rag_pipeline())
```

## 5.3 Advanced RAG Patterns with MCP {#advanced-rag-patterns-with-mcp}

### 5.3.1 Multi-Source RAG {#multi-source-rag}

```python
class MultiSourceRAG:
    """
    RAG system that retrieves from multiple MCP sources
    """
    
    async def query_multiple_sources(self, question: str):
        """Query across vector DB, web search, and databases"""
        
        # Parallel retrieval from multiple MCP servers
        results = await asyncio.gather(
            # Vector database
            self.vector_db_client.call_tool(
                "semantic_search",
                {"query": question, "n_results": 3}
            ),
            # Web search MCP server
            self.web_search_client.call_tool(
                "search",
                {"query": question, "num_results": 3}
            ),
            # SQL database MCP server
            self.database_client.call_tool(
                "semantic_query",
                {"question": question}
            )
        )
        
        # Combine and rank all results
        combined_context = self.combine_contexts(results)
        
        # Generate response
        return await self.generate_response(question, combined_context)
```

### 5.3.2 RAG with Re-ranking {#rag-with-re-ranking}

```python
class ReRankingRAG:
    """RAG with result re-ranking for better relevance"""
    
    async def query_with_rerank(self, question: str):
        """Retrieve, re-rank, then generate"""
        
        # Initial retrieval (over-fetch)
        initial_results = await self.vector_db_client.call_tool(
            "semantic_search",
            {"query": question, "n_results": 20}
        )
        
        # Re-rank using LLM
        reranked = await self.rerank_results(
            question,
            json.loads(initial_results.content[0].text)["results"]
        )
        
        # Use top results for generation
        top_results = reranked[:5]
        context = "\\n---\\n".join(doc["text"] for doc in top_results)
        
        # Generate final answer
        return await self.generate_response(question, context)
    
    async def rerank_results(self, question: str, results: list):
        """Use LLM to re-rank results by relevance"""
        # Ask LLM to score each result
        scores = []
        
        for result in results:
            response = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": f"""Rate the relevance of this text to the question on a scale of 0-10.
Only respond with a number.

Question: {question}

Text: {result['text'][:500]}

Relevance score:"""
                }]
            )
            
            score = float(response.content[0].text.strip())
            result["rerank_score"] = score
        
        # Sort by rerank score
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results
```

## 5.4 Summary {#summary}

MCP-enhanced RAG provides:

- ✅ **Standardized Data Access**: Unified interface to diverse sources
- ✅ **Multi-Source Retrieval**: Combine vector DB, web search, databases
- ✅ **Real-time Updates**: Subscribe to document changes
- ✅ **Flexible Processing**: Document parsing and chunking as MCP services
- ✅ **Advanced Patterns**: Re-ranking, hybrid search, conversational RAG

This architecture makes RAG systems more powerful, maintainable, and extensible.