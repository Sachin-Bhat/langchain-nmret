# --- Imports ---
import time
import uuid
from collections import namedtuple
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

# Import for metadata filtering
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import StringPromptValue
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from torch.utils._pytree import tree_map

# --- Titans PyTorch Import ---
try:
    # Import directly from the installed library or the local path if cloned
    from titans_pytorch.neural_memory import NeuralMemory, NeuralMemState, mem_state_detach

    print("Successfully imported NeuralMemory from titans_pytorch")
    _import_successful = True
except ImportError:
    print("ERROR: titans-pytorch not found or NeuralMemory components not importable.")
    print("Please ensure the library is installed (`pip install titans-pytorch`) or added to PYTHONPATH.")
    _import_successful = False

    # Define dummy classes if import fails, to allow code structure check
    class _DummyNeuralMemory(nn.Module):
        def __init__(self, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> None:
            super().__init__()
            print("WARN: Using Dummy NeuralMemory")

        def forward(
            self, *args: tuple[Any, ...], **kwargs: dict[str, Any]
        ) -> tuple[torch.Tensor, None, tuple[None, None]]:
            return torch.randn(1, 1, 384), None, (None, None)  # Dummy output

        def init_weights(self, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> None:
            return None

    _DummyNeuralMemState = namedtuple(
        "_DummyNeuralMemState", ["seq_index", "weights", "cache_store_segment", "states", "updates"]
    )

    def _dummy_mem_state_detach(state: _DummyNeuralMemState) -> _DummyNeuralMemState:
        return state  # Dummy

    print("Defined dummy NeuralMemory components.")

# Conditionally assign names
if not _import_successful:
    NeuralMemory = _DummyNeuralMemory  # type: ignore
    NeuralMemState = _DummyNeuralMemState  # type: ignore
    mem_state_detach = _dummy_mem_state_detach  # type: ignore


# --- Wrapper for Titans Neural Memory using the imported class ---
class TitansNeuralMemoryWrapper:
    """
    Wrapper around the titans-pytorch NeuralMemory that uses its
    internal state management and forward pass for updates and retrieval.
    Manages the NeuralMemState across calls.
    """

    def __init__(
        self, embedding_dim: int, device: str = "cpu", momentum: bool = True, **kwargs: dict[str, Any]
    ) -> None:
        """
        Initializes the wrapper and the underlying NeuralMemory model.

        Args:
            embedding_dim: The dimension of the embeddings used.
            device: The device ('cpu' or 'cuda') for PyTorch tensors.
            momentum: Whether the internal NeuralMemory should use momentum.
            **kwargs: Additional keyword arguments passed directly to NeuralMemory.
        """
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim
        self.use_momentum = momentum

        # Combine mandatory dim with other kwargs
        nm_kwargs = {"dim": embedding_dim, "momentum": self.use_momentum, **kwargs}

        # Instantiate the actual NeuralMemory model
        try:
            self.model = NeuralMemory(**nm_kwargs).to(self.device)  # type: ignore
        except Exception as e:
            print(f"ERROR: Failed to instantiate NeuralMemory with kwargs {nm_kwargs}. Error: {e}")
            raise e

        # State is managed internally by passing NeuralMemState objects
        self.current_state: NeuralMemState | None = None
        print(f"TitansNeuralMemoryWrapper: Initialized using NeuralMemory on {self.device}")
        print(f"  NeuralMemory Args used: {nm_kwargs}")

    def _ensure_state(self, batch_size: int) -> None:
        """Initializes state if it doesn't exist for the given batch size."""
        if self.current_state is None:
            print("TitansNeuralMemoryWrapper: Initializing internal NeuralMemState.")
            with torch.no_grad():
                init_w_raw = self.model.init_weights(batch_size)
                init_w = init_w_raw.detach()

                # Use the stored momentum flag
                init_mom_raw = self.model.init_momentum(batch_size) if self.use_momentum else None
                init_mom = init_mom_raw.detach() if init_mom_raw is not None else None

                # Ensure initial state tensors are on the correct device
                init_w = init_w.to(self.device) if init_w is not None else None
                init_mom = init_mom.to(self.device) if init_mom is not None else None

                # Initial past_state tuple (last_update_state, last_momentum_state)
                # Start with the initial weights as the 'last update' state
                initial_past_state = (init_w, init_mom)
                self.current_state = NeuralMemState(0, init_w, None, initial_past_state, None)

    @torch.no_grad()
    def retrieve(self, query_key_embedding: np.ndarray) -> np.ndarray:
        """Retrieve abstract guidance vector using the neural memory's forward pass."""
        self.model.eval()
        if not hasattr(self.model, "forward"):  # Check if model loaded correctly
            print("ERROR: NeuralMemory model not correctly initialized in wrapper.")
            return np.array([]).reshape(0, self.embedding_dim)

        if query_key_embedding.size == 0:
            return np.array([]).reshape(0, self.embedding_dim)

        batch_size = query_key_embedding.shape[0]
        self._ensure_state(batch_size)  # Ensure state is initialized

        query_tensor = torch.from_numpy(query_key_embedding).float().to(self.device)
        if query_tensor.ndim == 2:
            query_tensor = query_tensor.unsqueeze(1)  # Add seq dim

        print(f"TitansNeuralMemoryWrapper: Retrieving with input shape {query_tensor.shape}")

        # Call forward for retrieval only
        # When return_surprises=False, forward likely returns only 2 values
        retrieved_tensor, next_state = self.model.forward(
            seq=query_tensor,
            store_seq=None,
            state=self.current_state,
            detach_mem_state=True,  # Detach state during retrieval
            return_surprises=False,
        )
        # NOTE: We do NOT update self.current_state during pure retrieval

        print(f"TitansNeuralMemoryWrapper: Retrieval output shape {retrieved_tensor.shape}")
        if retrieved_tensor.shape[1] == 1:
            retrieved_tensor = retrieved_tensor.squeeze(1)
        # Explicitly cast to ndarray for type checker
        return np.asarray(retrieved_tensor.cpu().numpy())

    def update(self, source_sequence_for_kv: np.ndarray) -> float:
        """
        Update the neural memory's internal state by processing the source sequence.
        The model internally derives keys/values/etc. from this sequence.
        Returns loss proxy.
        """
        self.model.train()
        if not hasattr(self.model, "forward"):
            print("ERROR: NeuralMemory model not correctly initialized in wrapper.")
            return 0.0

        if source_sequence_for_kv.size == 0:
            print("TitansNeuralMemoryWrapper: No source data for update.")
            return 0.0

        batch_size = source_sequence_for_kv.shape[0]
        self._ensure_state(batch_size)  # Ensure state is initialized

        store_seq_tensor = torch.from_numpy(source_sequence_for_kv).float().to(self.device)
        if store_seq_tensor.ndim == 2:
            store_seq_tensor = store_seq_tensor.unsqueeze(1)

        print(f"TitansNeuralMemoryWrapper: Updating state with store_seq shape {store_seq_tensor.shape}")

        # Call forward for storage only.
        # Pass a dummy seq tensor to satisfy the forward method's signature
        # and avoid the internal ndim error when seq is None.
        batch_size = store_seq_tensor.shape[0]
        dummy_seq_tensor = torch.empty((batch_size, 0, self.embedding_dim), device=self.device)

        # Crucially, pass the current state to continue the sequence
        _, next_state, surprises = self.model.forward(
            seq=dummy_seq_tensor,  # Pass the dummy tensor
            store_seq=store_seq_tensor,
            state=self.current_state,  # Pass the managed state
            detach_mem_state=False,  # Allow state updates
            return_surprises=True,  # Get loss for surprise proxy
        )

        # Update the wrapper's persistent state for the next call
        self.current_state = next_state

        # Calculate average loss from surprises tuple (loss, lr)
        loss_tensor = surprises[0]
        loss_value = loss_tensor.mean().item() if loss_tensor is not None else 0.0

        print(f"TitansNeuralMemoryWrapper: Updated state. Loss proxy: {loss_value:.4f}")
        return loss_value

    def get_state(self) -> dict[str, Any]:
        """Get model parameters and current NeuralMemState for persistence."""
        # Need to handle device placement for state saving/loading if needed
        state_to_save = mem_state_detach(self.current_state) if self.current_state else None
        # Convert potentially complex state (TensorDicts) to basic types if necessary for saving
        # state_to_save = _convert_state_to_savable(state_to_save)
        return {"model_state_dict": self.model.state_dict(), "current_neural_mem_state": state_to_save}

    def load_state(self, state: dict[str, Any]) -> None:
        """Load model parameters and current NeuralMemState."""
        self.model.load_state_dict(state["model_state_dict"])
        # Convert loaded state back to tensors/TensorDicts if needed
        # loaded_mem_state = _convert_savable_to_state(state['current_neural_mem_state'])
        loaded_mem_state = state["current_neural_mem_state"]
        self.current_state = loaded_mem_state
        self.model.to(self.device)
        # Ensure loaded state tensors are on the correct device
        if self.current_state:
            # Move tensors to correct device
            self.current_state = tree_map(
                lambda t: t.to(self.device) if torch.is_tensor(t) else t, self.current_state
            )
            # Also handle TensorDict device placement if applicable
            # self.current_state = (
            #     self.current_state.to(self.device)
            #     if hasattr(self.current_state, 'to')
            #     else self.current_state
            # )

        print("TitansNeuralMemoryWrapper: Loaded state.")


# --- Contextual Memory using VectorStore (Remains the same) ---
class VectorStoreContextualMemory:
    """Uses a VectorStore to store and retrieve recent contextual information..."""

    def __init__(self, vectorstore: VectorStore, embedding_model: Embeddings) -> None:
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        print(f"VectorStoreContextualMemory: Initialized with VectorStore type: {type(vectorstore)}")

    def add(self, texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        if not texts:
            return

        # Use the metadatas directly, but sanitize them first for Chroma compatibility
        sanitized_metadatas = [_sanitize_metadata_for_chroma(meta) for meta in metadatas]

        # Ensure memory_id exists after sanitization
        for meta in sanitized_metadatas:
            if "memory_id" not in meta or not meta["memory_id"]:
                meta["memory_id"] = str(uuid.uuid4())

        try:
            # Use the sanitized metadata list
            ids = [m["memory_id"] for m in sanitized_metadatas]
            if hasattr(self.vectorstore, "add_embeddings"):
                self.vectorstore.add_embeddings(
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=sanitized_metadatas,  # Use sanitized metadata
                    ids=ids,
                )
            else:
                self.vectorstore.add_texts(
                    texts=texts, metadatas=sanitized_metadatas, ids=ids
                )  # Use sanitized metadata
            print(f"VectorStoreContextualMemory: Added {len(texts)} items using embeddings.")
        except (NotImplementedError, AttributeError, TypeError) as e1:
            print(
                f"VectorStoreContextualMemory: add_embeddings failed/not found/"
                f"wrong args ({e1}). Falling back to add_texts."
            )
            try:  # Try adding with IDs if add_texts supports it
                ids = [m["memory_id"] for m in sanitized_metadatas]
                self.vectorstore.add_texts(
                    texts=texts, metadatas=sanitized_metadatas, ids=ids
                )  # Use sanitized metadata
            except TypeError as e2:  # Fallback to adding without IDs
                print(
                    f"VectorStoreContextualMemory: add_texts with IDs failed ({e2}). "
                    "Falling back without IDs."
                )
                self.vectorstore.add_texts(
                    texts=texts, metadatas=sanitized_metadatas
                )  # Use sanitized metadata

    def retrieve_associative(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> list[tuple[Document, float]]:
        if query_embedding.size == 0:
            return []

        query_list = query_embedding.flatten().tolist()
        print(
            f"VectorStoreContextualMemory: Retrieving associatively "
            f"(top_k={top_k}) via vector similarity search."
        )
        try:
            # --- Try similarity_search_by_vector first ---
            docs = self.vectorstore.similarity_search_by_vector(
                embedding=query_list,
                k=top_k,
            )
            # Assign dummy scores as this method doesn't usually return them
            results_with_scores = [(doc, 0.0) for doc in docs]
            print(
                f"VectorStoreContextualMemory: Retrieved {len(results_with_scores)} "
                "items via similarity_search_by_vector."
            )
            return results_with_scores
        except Exception as e1:
            print(f"ERROR: similarity_search_by_vector failed: {e1}")
            # --- Fallback to similarity_search_with_relevance_scores ---
            print("VectorStoreContextualMemory: Falling back to similarity_search_with_relevance_scores.")
            try:
                results_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                    query="",  # Still might need a dummy query string
                    embedding=query_list,
                    k=top_k,
                )
                print(
                    f"VectorStoreContextualMemory: Retrieved {len(results_with_scores)} "
                    "items via similarity_search_with_relevance_scores."
                )
                return results_with_scores
            except Exception as e2:
                print(f"ERROR: Fallback similarity_search_with_relevance_scores also failed: {e2}")
                # Check if the error is the specific one about the 'embedding' keyword
                if "unexpected keyword argument 'embedding'" in str(e2):
                    print(
                        "  Hint: This might indicate an API mismatch between "
                        "langchain-chroma and chromadb versions."
                    )
                return []


# --- Helper function to sanitize metadata for Chroma ---
def _sanitize_metadata_for_chroma(metadata: dict) -> dict:
    sanitized_meta = {}

    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool):
            sanitized_meta[key] = value
        elif isinstance(value, list):
            # Convert list to a comma-separated string
            try:
                sanitized_meta[key] = ", ".join(map(str, value))
            except Exception:
                # Fallback if conversion fails
                sanitized_meta[key] = "[List Conversion Failed]"
        elif value is None:
            sanitized_meta[key] = ""  # Replace None with empty string
        else:
            # For other complex types, convert to string representation
            try:
                sanitized_meta[key] = str(value)
            except Exception:
                sanitized_meta[key] = "[Complex Type Conversion Failed]"
    return sanitized_meta


# --- LightThinker ---
class LightThinkerCompressor:
    """
    Compresses context (e.g., LLM thoughts) using summarization, inspired by LightThinker.
    Instead of hidden state manipulation, it generates a textual summary and its embedding.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,  # Use proper type from langchain_core
        embedding_model: Embeddings,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> None:
        self.llm = llm
        self.embedding_model = embedding_model
        self._compressed_state: np.ndarray | None = None
        print("LightThinkerCompressor: Initialized with LLM and Embedding Model.")

    def get_compressed_representation(self) -> np.ndarray | None:
        """Returns the current compressed state if available."""
        return self._compressed_state

    def compress(self, text_to_compress: str) -> tuple[str, np.ndarray]:
        """
        Compresses the input text by generating a concise summary using the LLM
        and computes the embedding of the summary.

        Args:
            text_to_compress: The string content (e.g., LLM thought) to compress.

        Returns:
            A tuple containing (summary_text, summary_embedding).
        """
        if not text_to_compress:
            return "", np.array([])

        print(f"LightThinkerCompressor: Compressing text (length: {len(text_to_compress)})...")
        prompt = (
            "You are part of a multi-step reasoning process addressing a user query. "
            "The following text is the output from the previous reasoning step. "
            "Your task is to create a concise summary of this step's key findings, "
            "conclusions, or actions, preserving the information most crucial for "
            "continuing the reasoning process. Focus on the core message that needs "
            "to be carried forward.\n\n"
            f"Previous Step's Output:\n---\n{text_to_compress}\n---\n\n"
            "Concise Summary for Next Step:"
        )

        try:
            summary_text = self.llm.invoke(prompt)
            summary_text = summary_text.strip().strip('"')
            print(f"  Generated Summary: {summary_text[:100]}...")
        except Exception as e:
            print(f"ERROR: LightThinker LLM summarization failed: {e}")
            summary_text = "[Compression Failed]"

        summary_embedding = np.array([])
        try:
            if summary_text and summary_text != "[Compression Failed]":
                embedding_result = self.embedding_model.embed_query(summary_text)
                if embedding_result:
                    summary_embedding = np.array(embedding_result)
                    self._compressed_state = summary_embedding
                    print(f"  Summary Embedding Shape: {summary_embedding.shape}")
        except Exception as e:
            print(f"ERROR: LightThinker summary embedding failed: {e}")

        return summary_text, summary_embedding


# --- Updated Custom Retriever (using new wrapper) ---
class NeuralMemoryRetriever(BaseRetriever):
    """Retriever integrating imported Titans NeuralMemory and VectorStore contextual memory."""

    vectorstore: VectorStore
    neural_memory: TitansNeuralMemoryWrapper  # Uses imported NeuralMemory now
    contextual_memory: VectorStoreContextualMemory
    compressor: LightThinkerCompressor
    llm: BaseLanguageModel
    embedding_model: Embeddings
    device: str = "cpu"

    # Parameters
    reasoning_steps: int = 1
    top_k_initial: int = 5
    top_k_contextual: int = 2
    compress_intermediate: bool = False
    update_memory_on_final: bool = False
    update_titans_with: str = "docs_and_llm"

    class Config:
        arbitrary_types_allowed = True

    def _get_embedding(self, text: str) -> np.ndarray:
        return np.array(self.embedding_model.embed_query(text)).reshape(1, -1)

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        if not texts:
            # Return shape (0, embedding_dim) if available, otherwise (0, 0)
            embedding_dim = getattr(self.neural_memory, "embedding_dim", 0)
            return np.array([]).reshape(0, embedding_dim)
        return np.array(self.embedding_model.embed_documents(texts))

    # _get_relevant_documents (Main logic loop - updates calls to neural_memory)
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        print(f"\n--- Starting Complex Retrieval for Query: '{query}' ---")
        all_context_docs: dict[str, Document] = {}
        reasoning_trace: list[str] = [f"Initial Query: {query}"]

        # 1. Initial Retrieval & Contextual Memory Add
        print(f"\nStep 1: Initial vector store retrieval (k={self.top_k_initial})")
        run_manager.on_text(f"Performing initial vector store search (k={self.top_k_initial}) for: {query}")
        initial_docs = self.vectorstore.similarity_search(query, k=self.top_k_initial)
        run_manager.on_text(f"Initial vector search found {len(initial_docs)} documents.")
        current_step_docs = []
        initial_doc_texts = []
        initial_doc_metadatas = []
        initial_doc_embeddings_list = []  # Need list of embeddings for contextual add

        # Process initial docs, converting dicts if necessary
        new_initial_docs = []
        for item in initial_docs:
            doc = None
            if isinstance(item, Document):
                doc = item
            elif isinstance(item, dict):  # type: ignore[unreachable]
                print(
                    f"WARN: Initial vector store retrieval returned dict: "
                    f"{str(item)[:100]}... Attempting conversion."
                )
                try:
                    page_content = item.get("page_content", "")
                    metadata = item.get("metadata", {})
                    doc = Document(page_content=page_content, metadata=metadata)
                except Exception as conversion_e:
                    print(f"WARN: Failed to convert dict to Document: {conversion_e}")
                    continue  # Skip if conversion fails
            else:
                print(f"WARN: Initial vector store retrieval returned non-Document/dict item: {type(item)}")
                continue  # Skip other types

            if doc is None:
                # Should not happen if logic above is correct
                continue  # type: ignore[unreachable]

            # Ensure metadata is a mutable dict if converted from dict
            # ideally shouldn't happen but a fallback nonetheless
            if not isinstance(doc.metadata, dict):
                doc.metadata = {}  # type: ignore[unreachable]

            doc_id = doc.metadata.get("memory_id", str(uuid.uuid4()))
            doc.metadata["memory_id"] = doc_id
            if doc_id not in all_context_docs:
                all_context_docs[doc_id] = doc
                new_initial_docs.append(doc)  # Add the processed Document
                # Ensure attributes exist before appending
                initial_doc_texts.append(getattr(doc, "page_content", ""))
                initial_doc_metadatas.append(getattr(doc, "metadata", {}))
        current_step_docs = new_initial_docs  # Update current_step_docs with only valid Documents

        if initial_doc_texts:
            # Get embeddings *once*
            initial_doc_embeddings_np = self._get_embeddings(initial_doc_texts)
            # Convert rows to list of lists/arrays
            initial_doc_embeddings_list = list(initial_doc_embeddings_np)
            # Add to contextual memory
            print("Step 1b: Adding initial docs to contextual memory...")
            run_manager.on_text(f"Adding {len(initial_doc_texts)} documents to contextual memory.")
            self.contextual_memory.add(initial_doc_texts, initial_doc_embeddings_list, initial_doc_metadatas)

        reasoning_trace.append(f"Initial retrieval added {len(current_step_docs)} new unique documents.")
        print(f"  Found {len(initial_docs)} docs, processed {len(current_step_docs)} unique Documents.")

        query_embedding = self._get_embedding(query)

        # --- Data collected for deferred TITANS memory update ---
        source_sequences_for_update: list[np.ndarray] = []

        # 2. Multi-Step Reasoning Loop
        for step in range(self.reasoning_steps):
            print(f"\n--- Reasoning Step {step + 1}/{self.reasoning_steps} ---")
            run_manager.on_text(f"Starting Reasoning Step {step + 1}/{self.reasoning_steps}")
            step_context_docs = current_step_docs

            # --- Prepare Input ---
            # Add try-except block to catch AttributeError and log the problematic item
            try:
                step_input_prompt, current_embeddings_for_memory = self._prepare_step_input(
                    query, reasoning_trace, step_context_docs, self.compressor.get_compressed_representation()
                )
            except AttributeError as ae:
                problematic_item = None
                for item in step_context_docs:
                    # Check if it's not a Document and lacks expected attributes
                    if not isinstance(item, Document):
                        try:  # type: ignore[unreachable]
                            _ = item.metadata  # Try accessing the attribute likely causing the error
                            # ideally shouldn't happen but a fallback nonetheless
                        except AttributeError:
                            problematic_item = item
                            break
                        try:
                            _ = item.page_content  # Also check page_content access
                        except AttributeError:
                            problematic_item = item
                            break

                print("ERROR: Caught AttributeError in _get_relevant_documents loop.")
                print(f"  Step: {step + 1}")
                if problematic_item is not None:
                    print(f"  Identified Problematic Item Type: {type(problematic_item)}")  # type: ignore[unreachable]
                    print(f"  Problematic Item Content (first 500 chars): {str(problematic_item)[:500]}")
                else:
                    print(
                        "  Could not isolate the specific problematic item, but an AttributeError occurred."
                    )
                print(f"  Full step_context_docs types at error: {[type(d) for d in step_context_docs]}")
                raise ae  # Re-raise the error after logging details

            # Calculate mean embedding for input
            step_input_embedding = (
                np.mean(current_embeddings_for_memory, axis=0).reshape(1, -1)
                if current_embeddings_for_memory.size > 0
                else query_embedding
            )

            # 2b. Retrieve from Contextual Memory (VectorStore)
            print(f"Step {step + 1}b: Contextual Memory Retrieval (k={self.top_k_contextual})")
            run_manager.on_text(f"Retrieving from contextual memory (k={self.top_k_contextual})")
            contextual_results = self.contextual_memory.retrieve_associative(
                step_input_embedding, top_k=self.top_k_contextual
            )
            reasoning_trace.append(f"Contextual memory retrieved {len(contextual_results)} items.")
            run_manager.on_text(f"Contextual memory retrieved {len(contextual_results)} items.")
            step_input_prompt += "\n\nContextual Memory Hints:\n"
            retrieved_context_docs = []
            for item, score in contextual_results:
                doc = None
                if isinstance(item, Document):
                    doc = item
                elif isinstance(item, dict):  # type: ignore[unreachable]
                    print(
                        f"WARN: Contextual memory retrieved dict: {str(item)[:100]}... Attempting conversion."
                    )
                    try:
                        page_content = item.get("page_content", "")
                        metadata = item.get("metadata", {})
                        doc = Document(page_content=page_content, metadata=metadata)
                    except Exception as conversion_e:
                        print(f"WARN: Failed to convert dict to Document: {conversion_e}")
                        continue
                else:
                    print(f"WARN: Contextual memory retrieved non-Document/dict item: {type(item)}")
                    continue

                # Should not happen if logic above is correct
                if doc is None:
                    continue  # type: ignore[unreachable]

                # Ensure metadata is a mutable dict if converted from dict
                # ideally shouldn't happen but a fallback nonetheless
                if not isinstance(doc.metadata, dict):
                    doc.metadata = {}  # type: ignore[unreachable]

                doc_id = doc.metadata.get("memory_id", "unknown_id")
                # Ensure metadata exists and add score before adding to prompt
                doc.metadata["retrieval_score"] = score  # Optionally add score to metadata

                content_preview = getattr(doc, "page_content", "")[:100]
                step_input_prompt += f"- (Score: {score:.3f}) ID: {doc_id}: {content_preview}...\n"
                if doc_id not in all_context_docs:
                    all_context_docs[doc_id] = doc
                    retrieved_context_docs.append(doc)  # Add the processed Document
            if retrieved_context_docs:
                print(f"  Added {len(retrieved_context_docs)} unique docs from contextual retrieval.")
                step_context_docs.extend(retrieved_context_docs)

            # 2c. Retrieve from Neural Memory (Titans Guidance)
            print(f"Step {step + 1}c: Neural Memory Retrieval")
            run_manager.on_text("Retrieving guidance from neural memory.")
            neural_memory_guidance = self.neural_memory.retrieve(step_input_embedding)
            reasoning_trace.append("Neural memory provided abstract guidance.")
            run_manager.on_text(f"Neural memory guidance vector shape: {neural_memory_guidance.shape}")
            step_input_prompt += "\nNeural Memory Guidance: [Utilized abstract guidance vector]"
            print(f"  Neural memory guidance vector shape: {neural_memory_guidance.shape}.")

            # 2d. Generate Next Thought / Refine Context (LLM Step)
            print(f"Step {step + 1}d: Generating next thought/action using LLM")
            llm_output = self.llm.invoke(step_input_prompt)
            reasoning_trace.append(f"LLM Output/Thought: {llm_output}")
            print(f"  LLM generated: '{llm_output[:150]}...'")
            # Log full before potential compression
            run_manager.on_text(f"LLM generated (full): {llm_output[:100]}...")

            # --- Compression Step (LightThinker Adaptation) ---
            compressed_llm_output_text = llm_output  # Default to original if no compression
            if self.compress_intermediate and self.compressor:
                print(f"Step {step + 1}e: Compressing LLM output using LightThinkerCompressor")
                run_manager.on_text(f"Compressing LLM output (length: {len(llm_output)})...")
                summary_text, summary_embedding = self.compressor.compress(llm_output)
                if summary_text != "[Compression Failed]":
                    compressed_llm_output_text = summary_text
                    # Replace last reasoning trace item (full LLM output) with the compressed version
                    reasoning_trace[-1] = f"LLM Output/Thought (Compressed): {compressed_llm_output_text}"
                    run_manager.on_text(f"LLM output compressed to: {compressed_llm_output_text[:100]}...")
                else:
                    run_manager.on_text("Compression failed, using original LLM output.")
            else:
                # Add the uncompressed thought to trace if not compressing
                reasoning_trace.append(f"LLM Output/Thought: {llm_output}")

            # --- Process LLM & Prepare Memory Update Data ---
            docs_for_next_step = step_context_docs  # Simplification: Update this based on LLM parsing
            current_step_docs = docs_for_next_step  # Set context for next loop / final output

            # Use the (potentially compressed) output for memory updates
            source_seq_this_step = self._prepare_source_sequence_for_update(
                step_context_docs, compressed_llm_output_text, self.update_titans_with
            )

            # --- Update Memories ---
            if source_seq_this_step is not None and source_seq_this_step.size > 0:
                # Update/Accumulate Titans data
                if not self.update_memory_on_final:
                    print(f"Step {step + 1}f: Updating Neural Memory (End of Step)")
                    # Pass source sequence
                    loss = self.neural_memory.update(source_sequence_for_kv=source_seq_this_step)
                    print(f"  Titans updated state. Surprise/Loss proxy: {loss:.4f}")
                    run_manager.on_text(
                        f"Updated neural memory state (end of step {step + 1}). Loss proxy: {loss:.4f}"
                    )
                else:
                    # Defer update
                    source_sequences_for_update.append(source_seq_this_step)

                # Always update Contextual Memory immediately
                # Use the (potentially compressed) output for contextual add
                keys_ctx, _, texts_ctx, metas_ctx = self._prepare_data_for_contextual_memory_add(
                    step_context_docs, compressed_llm_output_text, self.update_titans_with
                )
                # Check if data was actually prepared before adding
                if keys_ctx is not None and texts_ctx is not None and metas_ctx is not None:
                    print(f"Step {step + 1}g: Updating Contextual Memory")
                    run_manager.on_text(f"Updating contextual memory with {len(texts_ctx)} items.")
                    # Convert NumPy embeddings to list of lists for vector store add
                    embeddings_list_ctx = list(map(list, keys_ctx))
                    self.contextual_memory.add(texts_ctx, embeddings_list_ctx, metas_ctx)
            # --- End of loop ---
            run_manager.on_text(f"Finished Reasoning Step {step + 1}/{self.reasoning_steps}")

        # 3. Final Neural Memory Update (if deferred)
        if self.update_memory_on_final and source_sequences_for_update:
            print("\n--- Final Neural Memory Update (End of Run) ---")
            if len(source_sequences_for_update) > 0:
                # Ensure all sequences have the same embedding dimension before concatenating
                first_dim = source_sequences_for_update[0].shape[-1]
                if all(s.shape[-1] == first_dim for s in source_sequences_for_update):
                    # Concatenate along sequence dimension (axis=1 assuming shape [1, n, d])
                    combined_source_seq = np.concatenate(source_sequences_for_update, axis=1)
                    loss = self.neural_memory.update(
                        source_sequence_for_kv=combined_source_seq
                    )  # Pass combined source
                    print(f"  Titans updated state. Final Surprise/Loss proxy: {loss:.4f}")
                    run_manager.on_text(
                        f"Updated neural memory state (end of run). Final loss proxy: {loss:.4f}"
                    )
                else:
                    print(
                        "ERROR: Cannot concatenate source sequences for "
                        "final update due to mismatched dimensions."
                    )
            else:
                print("  No source sequences collected for final update.")

        # 4. Final Document Selection
        print("\n--- Finalizing Retrieval ---")
        final_documents = current_step_docs

        # === DEBUGGING: Check types before returning ===
        print("--- DEBUG: Pre-filtering final_documents ---")
        print(f"Type: {type(final_documents)}, Length: {len(final_documents)}")
        for idx, item in enumerate(final_documents):
            print(f"Item {idx} type: {type(item)}")
        print("--- DEBUG: End pre-filtering ---")
        # === END DEBUGGING ===

        # Explicitly filter the final_documents list to ensure only Document instances are included
        final_document_objects = [doc for doc in final_documents if isinstance(doc, Document)]

        if len(final_document_objects) != len(final_documents):
            filtered_count = len(final_documents) - len(final_document_objects)
            print(f"WARN: Filtered out {filtered_count} non-Document items before returning.")

        print(f"Returning {len(final_document_objects)} actual Document objects from the retriever.")

        # Log retriever end with the filtered list
        run_manager.on_retriever_end(final_document_objects)

        return final_document_objects  # Return the filtered list

    # --- Helper Methods ---
    def _prepare_step_input(
        self,
        original_query: str,
        reasoning_trace: list[str],
        context_docs: list[Document],
        compressed_state: np.ndarray | None,
    ) -> tuple[str, np.ndarray]:
        """Prepare input for the next reasoning step."""
        prompt = f"Original Query: {original_query}\n\nRecent Reasoning History:\n"
        prompt += "\n".join([f"- {t}" for t in reasoning_trace[-3:]])

        if compressed_state is not None:
            prompt += "\n\nCompressed Context State Available."

        prompt += "\n\nCurrent Context Documents:\n"

        # Filter context_docs to ensure they are Documents and get embeddings safely
        valid_context_docs = [doc for doc in context_docs if isinstance(doc, Document)]
        doc_embeddings = self._get_embeddings([getattr(d, "page_content", "") for d in valid_context_docs])

        if valid_context_docs:
            for i, doc in enumerate(valid_context_docs):
                # Safely access metadata and page_content
                doc_meta = getattr(doc, "metadata", {})
                doc_id = doc_meta.get("memory_id", "N/A")
                content_preview = getattr(doc, "page_content", "")[:150]
                prompt += f"- Doc {i + 1} (ID: {doc_id}): {content_preview}...\n"
        else:
            prompt += "- None\n"

        prompt += (
            "\nTask: Analyze history, hints, and docs. "
            "Generate reasoning or identify next steps. "
            "Utilize neural memory guidance."
        )
        return prompt, doc_embeddings

    def _prepare_source_sequence_for_update(
        self, docs: list[Document], llm_output: str, mode: str
    ) -> np.ndarray | None:
        """Prepares the single sequence input required by NeuralMemory.forward for storage."""
        print(f"  _prepare_source_sequence_for_update (Mode: {mode})")
        source_texts = []
        valid_docs = [doc for doc in docs if isinstance(doc, Document)]  # Filter upfront

        if mode in ["docs_only", "docs_and_llm"] and valid_docs:
            source_texts.extend([getattr(d, "page_content", "") for d in valid_docs])

        if mode in ["llm_only", "docs_and_llm"] and llm_output:
            source_texts.append(llm_output)

        if not source_texts:
            return None

        source_embeddings = self._get_embeddings(source_texts)  # Shape: (num_items, dim)
        # Shape for NeuralMemory.forward store_seq: (batch, seq, dim)
        # Ensure correct embedding dimension before reshape
        default_dim = source_embeddings.shape[-1] if source_embeddings.size > 0 else 0
        embedding_dim = getattr(self.neural_memory, "embedding_dim", default_dim)
        if source_embeddings.size == 0:
            return None  # Return None if no embeddings generated
        return source_embeddings.reshape(1, -1, embedding_dim)

    def _prepare_data_for_contextual_memory_add(
        self, docs: list[Document], llm_output: str, mode: str
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[str] | None, list[dict] | None]:
        """Gets embeddings, texts, and metadata suitable for VectorStoreContextualMemory.add"""
        print(f"  _prepare_data_for_contextual_memory_add (Mode: {mode})")
        texts_ctx = []
        metas_ctx = []
        valid_docs = [doc for doc in docs if isinstance(doc, Document)]  # Filter upfront

        if mode in ["docs_only", "docs_and_llm"] and valid_docs:
            texts_ctx.extend([getattr(d, "page_content", "") for d in valid_docs])
            # Crucially, ensure metadata includes the 'memory_id' used by the vector store
            # Use getattr for metadata access and ensure it's a dict
            current_metas = []
            for d in valid_docs:
                meta = getattr(d, "metadata", {})
                if not isinstance(meta, dict):  # Ensure it's a dict
                    meta = {}
                meta_copy = meta.copy()
                if "memory_id" not in meta_copy:
                    meta_copy["memory_id"] = str(uuid.uuid4())
                current_metas.append(meta_copy)
            metas_ctx.extend(current_metas)

        if mode in ["llm_only", "docs_and_llm"] and llm_output:
            texts_ctx.append(llm_output)
            metas_ctx.append(
                {"memory_id": str(uuid.uuid4()), "type": "llm_thought", "timestamp": time.time()}
            )

        if not texts_ctx:
            return None, None, None, None

        embeddings = self._get_embeddings(texts_ctx)
        return embeddings, None, texts_ctx, metas_ctx


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Initializing Functional Components ---")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun, Callbacks
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_core.outputs import Generation, LLMResult
    from langchain_core.prompt_values import PromptValue, StringPromptValue
    from langchain_core.runnables import RunnableConfig

    # 1. Embedding Model
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
        )
        EMBEDDING_DIM = 384
    except Exception as e:
        print(f"HF Embeddings Error: {e}")
        exit()

    # 2. Vector Store
    try:
        vectorstore = Chroma(
            collection_name="nmret_db_v1",
            embedding_function=embedding_model,
            persist_directory="./chroma_db_persist_v1",
        )
        vectorstore.add_texts(
            ["Mosaics uses kernels.", "Titans learns online.", "LightThinker is conceptual."],
            metadatas=[{"source": "init", "memory_id": f"init_{i}"} for i in range(3)],
        )
        print("Initialized Chroma vector store.")
    except Exception as e:
        print(f"Chroma Error: {e}")
        exit()

    # 3. Titans Wrapper with NeuralMemory kwargs
    # Ensure these match expectations of the imported NeuralMemory
    nm_kwargs_example = {
        "layers": 1,  # Depth of the memory MLP/Attn model
        "heads": 2,  # Number of heads for internal projections/norms
        "chunk_size": 256,  # Chunk size for processing updates
        "momentum": True,  # Enable momentum
        "qk_rmsnorm": False,  # Example: Disable QK norm
    }
    titans_wrapper = TitansNeuralMemoryWrapper(embedding_dim=EMBEDDING_DIM, device=DEVICE)

    # 4. Contextual Memory Wrapper
    contextual_memory = VectorStoreContextualMemory(vectorstore=vectorstore, embedding_model=embedding_model)

    # 5. LLM (Dummy)
    class DummyRunnable(BaseLanguageModel):
        """Mock LLM for testing purposes."""

        def invoke(
            self,
            input: LanguageModelInput,  # noqa: A002
            config: RunnableConfig | None = None,
            **kwargs: dict[str, Any],
        ) -> str:
            prompt = str(input)  # Convert input to string
            print(f"DummyLLM invoke called (prompt length: {len(prompt)})")
            time.sleep(0.05)
            return f"LLM reasoned: {prompt.split()[-1]}"

        async def ainvoke(
            self,
            input: LanguageModelInput,  # noqa: A002
            config: RunnableConfig | None = None,
            **kwargs: dict[str, Any],
        ) -> str:
            return self.invoke(input, config, **kwargs)

        def _generate(
            self,
            prompts: list[str],
            stop: list[str] | None = None,
            run_managers: list[CallbackManagerForLLMRun] | None = None,
            **kwargs: dict[str, Any],
        ) -> LLMResult:
            # _generate receives run managers (one per prompt) from generate_prompt
            generations = []
            for i, prompt in enumerate(prompts):
                # Perform dummy generation
                text = f"Dummy response to: {prompt[:50]}..."
                gen = [Generation(text=text)]
                generations.append(gen)
                # Get the specific run manager for this prompt
                llm_run_manager = run_managers[i] if run_managers else None
                # Call on_llm_end for this specific run
                if llm_run_manager:
                    llm_run_manager.on_llm_end(LLMResult(generations=[gen]))

            return LLMResult(generations=generations)

        async def _agenerate(
            self,
            prompts: list[str],
            stop: list[str] | None = None,
            run_managers: list[AsyncCallbackManagerForLLMRun] | None = None,
            **kwargs: dict[str, Any],
        ) -> LLMResult:
            # _agenerate receives run managers (one per prompt) from agenerate_prompt
            generations = []
            for i, prompt in enumerate(prompts):
                # Perform dummy generation
                text = f"Dummy async response to: {prompt[:50]}..."
                gen = [Generation(text=text)]
                generations.append(gen)
                # Get the specific run manager for this prompt
                llm_run_manager = run_managers[i] if run_managers else None
                # Call on_llm_end for this specific run
                if llm_run_manager:
                    await llm_run_manager.on_llm_end(LLMResult(generations=[gen]))

            return LLMResult(generations=generations)

        # --- Correct implementations for generate_prompt and agenerate_prompt ---
        def generate_prompt(
            self,
            prompts: list[PromptValue],
            stop: list[str] | None = None,
            callbacks: Callbacks = None,
            **kwargs: dict[str, Any],
        ) -> LLMResult:
            # Configure the top-level manager
            callback_manager = CallbackManager.configure(callbacks, self.callbacks, self.verbose)
            prompt_strings = [str(p) for p in prompts]
            # Call on_llm_start to get list of run managers
            run_managers = callback_manager.on_llm_start(
                self.model_dump(), prompt_strings, invocation_params=kwargs, options=self.model_dump()
            )
            # Call _generate with the list of run managers
            results = self._generate(prompt_strings, stop=stop, run_managers=run_managers, **kwargs)
            return results

        async def agenerate_prompt(
            self,
            prompts: list[PromptValue],
            stop: list[str] | None = None,
            callbacks: Callbacks = None,
            **kwargs: dict[str, Any],
        ) -> LLMResult:
            # Configure the top-level manager
            callback_manager = AsyncCallbackManager.configure(callbacks, self.callbacks, self.verbose)
            prompt_strings = [str(p) for p in prompts]
            # Call on_llm_start to get list of run managers
            run_managers = await callback_manager.on_llm_start(
                self.model_dump(), prompt_strings, invocation_params=kwargs, options=self.model_dump()
            )
            # Call _agenerate with the list of run managers
            results = await self._agenerate(prompt_strings, stop=stop, run_managers=run_managers, **kwargs)
            return results

        def predict(
            self,
            text: str,
            *,
            stop: Sequence[str] | None = None,
            **kwargs: dict[str, Any],
        ) -> str:
            # predict should call generate_prompt
            callbacks = kwargs.pop("callbacks", None)
            prompt_value = StringPromptValue(text=text)
            result = self.generate_prompt(
                [prompt_value],
                stop=list(stop) if stop else None,
                callbacks=cast(Callbacks, callbacks),
                **kwargs,
            )
            return result.generations[0][0].text

        def predict_messages(
            self,
            messages: list[BaseMessage],
            *,
            stop: Sequence[str] | None = None,
            **kwargs: dict[str, Any],
        ) -> BaseMessage:
            # predict_messages should call generate_prompt
            callbacks = kwargs.pop("callbacks", None)
            # Simplistic conversion for dummy: join message content
            prompt_text = "\n".join([str(m.content) for m in messages])
            prompt_value = StringPromptValue(text=prompt_text)
            result = self.generate_prompt(
                [prompt_value],
                stop=list(stop) if stop else None,
                callbacks=cast(Callbacks, callbacks),
                **kwargs,
            )
            return HumanMessage(content=result.generations[0][0].text)

        async def apredict(
            self,
            text: str,
            *,
            stop: Sequence[str] | None = None,
            **kwargs: dict[str, Any],
        ) -> str:
            # apredict should call agenerate_prompt
            callbacks = kwargs.pop("callbacks", None)
            prompt_value = StringPromptValue(text=text)
            result = await self.agenerate_prompt(
                [prompt_value],
                stop=list(stop) if stop else None,
                callbacks=cast(Callbacks, callbacks),
                **kwargs,
            )
            return result.generations[0][0].text

        async def apredict_messages(
            self,
            messages: list[BaseMessage],
            *,
            stop: Sequence[str] | None = None,
            **kwargs: dict[str, Any],
        ) -> BaseMessage:
            # apredict_messages should call agenerate_prompt
            callbacks = kwargs.pop("callbacks", None)
            # Simplistic conversion for dummy: join message content
            prompt_text = "\n".join([str(m.content) for m in messages])
            prompt_value = StringPromptValue(text=prompt_text)
            result = await self.agenerate_prompt(
                [prompt_value],
                stop=list(stop) if stop else None,
                callbacks=cast(Callbacks, callbacks),
                **kwargs,
            )
            return HumanMessage(content=result.generations[0][0].text)

        @property
        def _llm_type(self) -> str:
            return "dummy"

    llm = DummyRunnable()

    # 6. LightThinker Compressor (Initialized with LLM and Embeddings)
    compressor = LightThinkerCompressor(llm=llm, embedding_model=embedding_model)

    # Create the retriever instance
    retriever = NeuralMemoryRetriever(
        vectorstore=vectorstore,
        neural_memory=titans_wrapper,
        contextual_memory=contextual_memory,
        compressor=compressor,
        llm=llm,
        embedding_model=embedding_model,
        device=DEVICE,
        reasoning_steps=2,  # More steps to see memory updates
        top_k_initial=3,
        top_k_contextual=2,
        compress_intermediate=False,
        update_memory_on_final=False,
        update_titans_with="docs_and_llm",
    )

    # Run a query
    print("\n\n--- Running Functional Retriever (using imported NeuralMemory) ---")
    try:
        query = "How does Titans update its memory?"
        results = retriever.invoke(query)
        print("\n--- Retrieval Complete ---")
        print(f"Final Documents Returned: {len(results)}")
        for i, doc in enumerate(results):
            doc_id = doc.metadata.get("memory_id", "N/A")
            content_preview = doc.page_content[:100]
            print(f"  - Doc {i}: ID: {doc_id}, Content: {content_preview}...")

    except Exception as e:
        print(f"\nFunctional execution failed: {e}")
        import traceback

        traceback.print_exc()

    print("\nNOTE: Run used functional VectorStore and the imported Titans NeuralMemory via wrapper.")
    print("NeuralMemory state updates are handled by its internal `forward` method.")
