from typing import Dict, Any
import logging
import json
import os
from typing import List,Dict, Any
from dotenv import load_dotenv
load_dotenv()
from io import StringIO
from contextlib import redirect_stdout
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
## =======================================
### Functions for Recomendation tool 
## =======================================
from llama_index.core.schema import Document
from llama_index.core import Settings
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import PromptTemplate
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser.text import SemanticSplitterNodeParser
from llama_index.llms.openai import OpenAI
# from sentence_transformers import SentenceTransformer
from llama_index.embeddings.openai import OpenAIEmbedding

# model = SentenceTransformer("BAAI/bge-small-en-v1.5")
# model.save("local_bge_model")

# Later:
embedding_model = OpenAIEmbedding(model="text-embedding-3-small")
# embedding_model = "BAAI/bge-small-en-v1.5"

# Set up OpenAI models (you can replace with HuggingFace if needed)
llm = OpenAI(model="gpt-4o-mini")
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=gpt_api_key)

# Use semantic node parser
parser = SemanticSplitterNodeParser(
    embed_model=embedding_model,
    llm=llm,
    breakpoint_percentile_threshold=95,  # higher = fewer chunks, more semantic
    buffer_size=1
)

## BM25 Retriever for recommendation tool
class WeightedHybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever, 
                 vector_weight=0.6, bm25_weight=0.4):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        super().__init__()
        print(f"WeightedHybridRetriever initialized with weights: Vector={vector_weight}, BM25={bm25_weight}")

    def _retrieve(self, query_str, **kwargs):
        print(f"\n=== WEIGHTED HYBRID RETRIEVAL PROCESS ===")
        print(f"Query: {query_str}")
        
        # Step 1: Retrieve from vector retriever
        print(f"\n--- Step 1: Vector Retrieval (Weight: {self.vector_weight}) ---")
        vector_nodes = self.vector_retriever.retrieve(query_str, **kwargs)
        print(f"Vector retriever found {len(vector_nodes)} nodes")
        for i, node in enumerate(vector_nodes):
            print(f"  Vector Node {i+1}: Score={node.score:.4f} (Original)")
        
        # Step 2: Retrieve from BM25 retriever  
        print(f"\n--- Step 2: BM25 Retrieval (Weight: {self.bm25_weight}) ---")
        bm25_nodes = self.bm25_retriever.retrieve(query_str, **kwargs)
        print(f"BM25 retriever found {len(bm25_nodes)} nodes")
        for i, node in enumerate(bm25_nodes):
            print(f"  BM25 Node {i+1}: Score={node.score:.4f} (Original)")
        
        # Step 3: Apply weights to scores
        print(f"\n--- Step 3: Applying Weights ---")
        for i, node in enumerate(vector_nodes):
            original_score = node.score
            node.score = node.score * self.vector_weight
            print(f"  Vector Node {i+1}: {original_score:.4f} -> {node.score:.4f} (weighted)")
            
        for i, node in enumerate(bm25_nodes):
            original_score = node.score
            node.score = node.score * self.bm25_weight
            print(f"  BM25 Node {i+1}: {original_score:.4f} -> {node.score:.4f} (weighted)")
        
        # Step 4: Combine and deduplicate
        print(f"\n--- Step 4: Combining and Deduplicating ---")
        all_nodes = []
        node_ids = set()
        node_source_map = {}  # Track which retriever each node came from
        
        # Add vector nodes first
        for node in vector_nodes:
            if node.node.node_id not in node_ids:
                all_nodes.append(node)
                node_ids.add(node.node.node_id)
                node_source_map[node.node.node_id] = "Vector"
            else:
                print(f"  Duplicate found in vector results: {node.node.node_id}")
        
        # Add BM25 nodes, handling duplicates
        for node in bm25_nodes:
            if node.node.node_id not in node_ids:
                all_nodes.append(node)
                node_ids.add(node.node.node_id)
                node_source_map[node.node.node_id] = "BM25"
            else:
                # Handle duplicate by combining scores or taking max
                existing_node = next(n for n in all_nodes if n.node.node_id == node.node.node_id)
                combined_score = existing_node.score + node.score
                print(f"  Duplicate found in BM25: {node.node.node_id}")
                print(f"    Combining scores: {existing_node.score:.4f} + {node.score:.4f} = {combined_score:.4f}")
                existing_node.score = combined_score
                node_source_map[node.node.node_id] = "Both"
        
        # Step 5: Sort by final weighted scores
        print(f"\n--- Step 5: Final Sorting ---")
        all_nodes.sort(key=lambda x: x.score, reverse=True)
        
        print(f"Final combined results ({len(all_nodes)} unique nodes):")
        for i, node in enumerate(all_nodes):
            source = node_source_map.get(node.node.node_id, "Unknown")
            print(f"  Rank {i+1}: Score={node.score:.4f}, Source={source}, ID={node.node.node_id[:8]}...")
        
        return all_nodes

def clean_transcripts(convo):
    keys_to_remove = ["utterance_id", "timestamp", "speaker", "text", "sentiment"]
    segments = convo.get("conversation_details", {}).get("transcript_segments", [])
    for segment in segments:
        for key in keys_to_remove:
            segment.pop(key, None)  # Use pop with default to avoid KeyError
    return convo

### removed the "utterance_id", "timestamp", "speaker", "text", "sentiment" keys from the transcript segments
### This function cleans the transcripts by removing unnecessary keys from each segment.
## the input is the collated json file 
def apply_cleaning_transcripts(json_data):
    if isinstance(json_data, dict):
        cleaned_data = [clean_transcripts(json_data)]
    elif isinstance(json_data, list):
        cleaned_data = [clean_transcripts(convo) for convo in json_data]
    return cleaned_data


def matches_criteria(convo, issue_keyword, status_required):
    summary = convo.get("conversation_details", {}).get("summary", {})
    primary_issue = summary.get("primary_issue", "")
    resolution_status = summary.get("resolution_status", "")

    return (
        issue_keyword.lower() in primary_issue.lower()
        and resolution_status.lower() == status_required.lower()
    )
    

### target issue is a issue from context, required status is  the status from the context 
    
## applies matching criteria and return filters convo data, collated json file the input is the cleaned file from apply_cleaning_transcripts function
def apply_matching_criteria(json_data, target_issue, required_status):
    if isinstance(json_data, dict):
        filtered = [json_data] if matches_criteria(json_data, target_issue, required_status) else []
    elif isinstance(json_data, list):
        filtered = [convo for convo in json_data if matches_criteria(convo, target_issue, required_status)]
    return filtered 

### function to take in the filtered data and return nodes 
# Wrap JSON into a Document
def create_nodes_from_json(json_data):
    doc = Document(text=json.dumps(json_data, indent=2))
    # Parse it
    nodes = Settings.node_parser.get_nodes_from_documents([doc])
    for node in nodes:
        node_embedding = embedding_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
    # print(nodes)
    return nodes 

## retreival function to take in the nodes and return the context 
def retrieve_context_from_nodes(nodes, customer_concern_summary):
    print("=== INITIALIZING RETRIEVERS ===")
    corpus_size = len(nodes)
    ### shows an error if node returns only 1 node
    max_top_k = min(3, len(nodes))
    print(f"Setting top_k to: {max_top_k}")

    # Create individual retrievers
    print("Setting up Vector Retriever...")
    # 4. Build the index with the custom embeddings
    index = VectorStoreIndex(nodes, embed_model=embedding_model)
    # vector_retriever = index.as_retriever(similarity_top_k=3)
    vector_retriever = index.as_retriever(similarity_top_k=max_top_k)

    print("Setting up BM25 Retriever...")
    # bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=max_top_k)

    # Create WeightedHybridRetriever
    print("Creating WeightedHybridRetriever...")
    weighted_hybrid_retriever = WeightedHybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        vector_weight=0.6,  # 60% weight for semantic similarity
        bm25_weight=0.4     # 40% weight for keyword matching
    )

    corpus_size = len(nodes)
    print(f"{corpus_size}")

    # =============================================================================
    # 3. RETRIEVAL PROCESS
    # =============================================================================

    # print("\n" + "="*60)
    print("\nSTARTING RETRIEVAL PROCESS")
    # print("="*60)

    # Perform retrieval
    retrieved_nodes = weighted_hybrid_retriever.retrieve(customer_concern_summary)

    print(f"\n=== RETRIEVAL COMPLETE ===")
    print(f"Total nodes retrieved: {len(retrieved_nodes)}")

    # =============================================================================
    # 4. NODE FORMATTING (Your requested snippet)
    # =============================================================================

    print(f"\n=== FORMATTING RETRIEVED NODES ===")

    formattedDocs = []
    retrieved_information = ""

    for index, node in enumerate(retrieved_nodes):        
        formattedDocs.append({
            f"node{index+1}": node.to_dict()['node']['text']         
        })

    formattedResult = { "retrieved_documents": formattedDocs }
    nextStrResult = json.dumps(formattedResult, indent=2)
    retrieved_information = nextStrResult
    print(retrieved_information)
    return retrieved_information





# ============================================================================
# Context manager for ReAct Agent state management
# ============================================================================
class AgentContext:
    """Context manager for ReAct Agent state management"""
    
    def __init__(self):
        self.state = {
            "transcript": None,
            "name": None,
            "issue": None,
            "summary": None,
            "status": None,
            "conversation_context":None,
            "transaction_context":None,
            "rootcause": None,
            "pattern": None,
            "timeline": None,
            "rationale": None,
            "needs_upgrade":False,
            "context_decision":False,
            "retrieved_information": None,
            "concern_query": None
        }
        self._initialized = False
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the context"""
        self.state[key] = value
        logging.info(f"Context updated: {key} = {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
    
    def get(self, key: str) -> Any:
        """Get a value from the context"""
        return self.state.get(key)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple values at once"""
        self.state.update(updates)
        logging.info(f"Context batch update: {list(updates.keys())}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all context state"""
        return self.state.copy()
    
    def initialize(self, transcript_content: str):
        """Initialize context with transcript"""
        if not self._initialized:
            self.set("transcript", transcript_content)
            self._initialized = True
            logging.info("Context initialized with transcript")
    
    def is_complete(self) -> bool:
        """Check if all required fields are populated"""
        required_fields = ["name", "issue", "summary", "status"]
        return all(self.state.get(field) is not None for field in required_fields)
    
    def get_missing_fields(self) -> List[str]:
        """Get list of missing required fields"""
        required_fields = ["name", "issue", "summary", "status"] 
        return [field for field in required_fields if self.state.get(field) is None]


from contextlib import redirect_stdout
import sys
import threading
import time
from io import StringIO

class StreamCapture:
    def __init__(self):
        self.buffer = StringIO()
        self.new_data = threading.Event()
        self.lock = threading.Lock()
        self.closed = False

    def write(self, text):
        with self.lock:
            self.buffer.write(text)
            self.new_data.set()

    def flush(self):
        pass

    def readlines(self):
        while not self.closed:
            self.new_data.wait(timeout=0.1)
            with self.lock:
                self.buffer.seek(0)
                lines = self.buffer.readlines()
                self.buffer = StringIO()
                self.new_data.clear()
            for line in lines:
                yield line

    def close(self):
        self.closed = True
        self.new_data.set()