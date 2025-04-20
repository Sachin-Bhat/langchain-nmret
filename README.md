# Langchain Neural Memory Retriever (langchain-nmret)

This project implements a custom LangChain retriever, `NeuralMemoryRetriever`, designed to integrate stateful neural memory capabilities with standard vector store retrieval and LLM-based reasoning compression. It leverages the `titans-pytorch` library for its core neural memory component.

## Overview

The `NeuralMemoryRetriever` combines several components to provide a more sophisticated retrieval mechanism:

1.  **Titans Neural Memory Wrapper (`TitansNeuralMemoryWrapper`)**: Manages an instance of `titans-pytorch.NeuralMemory`, handling its state and providing methods to update the memory with new sequences and retrieve abstract guidance vectors based on query embeddings.
2.  **Vector Store Contextual Memory (`VectorStoreContextualMemory`)**: Uses a standard LangChain `VectorStore` (e.g., Chroma) to store and retrieve recent, concrete contextual information based on semantic similarity.
3.  **LightThinker Compressor (`LightThinkerCompressor`)**: An optional component inspired by the LightThinker concept. It uses a provided LangChain `BaseLanguageModel` to summarize intermediate LLM thoughts or outputs, creating a compressed textual representation and its corresponding embedding.
4.  **Multi-Step Retrieval Process**: Executes a configurable number of reasoning steps. Each step can involve:
    *   Retrieving abstract guidance from the Titans Neural Memory.
    *   Retrieving relevant recent context from the Vector Store Contextual Memory.
    *   Generating an intermediate thought or refined query using a `BaseLanguageModel`.
    *   Optionally compressing the LLM output using the `LightThinkerCompressor`.
    *   Updating both the neural memory and contextual memory based on the step's activities.

## Features

*   Integrates stateful, abstract neural memory (`titans-pytorch`) with standard vector retrieval.
*   Maintains recent context using a `VectorStore`.
*   Optional LLM-based compression of intermediate reasoning steps.
*   Configurable multi-step reasoning loop.
*   Flexible memory update strategies (e.g., update neural memory per step or only at the end).
*   Built on LangChain core interfaces (`BaseRetriever`, `BaseLanguageModel`, `Embeddings`, `VectorStore`).

## Installation

This project is available on PyPI.

```bash
pip install langchain-nmret
```

## Development Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd langchain-nmret
    ```

2.  **Install dependencies:**
    This project uses `uv` for package management. Ensure `uv` is installed (`pip install uv`).
    ```bash
    uv pip install -r requirements.txt # Or uv sync if using pyproject.toml dependencies directly
    ```
    Key dependencies include:
    *   `langchain-core`
    *   `titans-pytorch` (Requires separate installation/setup if not on PyPI or requires specific version)
    *   `torch`
    *   `numpy`
    *   A `VectorStore` implementation (e.g., `langchain-chroma`)
    *   An `Embeddings` implementation (e.g., `langchain-community`, `sentence-transformers`)
    *   A `BaseLanguageModel` implementation (e.g., `langchain-openai`, `langchain-huggingface`)

## Usage Example

```python
import time
import uuid
import numpy as np
import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

# --- Assume necessary imports from the package ---
from src.langchain_nmret.nmret import (
    NeuralMemoryRetriever,
    TitansNeuralMemoryWrapper,
    VectorStoreContextualMemory,
    LightThinkerCompressor,
)

# --- Mock/Dummy Components (Replace with actual implementations) ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.language_models.base import BaseLanguageModel # For Dummy LLM type hint
from langchain_core.outputs import Generation, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun, Callbacks
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.runnables import RunnableConfig

# Dummy LLM (Replace with OpenAI, HuggingFace, etc.)
class DummyRunnable(BaseLanguageModel):
    def _generate(self, prompts: list[str], stop: list[str] | None = None, run_managers: list[CallbackManagerForLLMRun] | None = None, **kwargs) -> LLMResult:
        generations = []
        for i, prompt in enumerate(prompts):
            text = f"Dummy response to: {prompt.split()[-1]}..."
            gen = [Generation(text=text)]
            generations.append(gen)
            if run_managers and run_managers[i]:
                run_managers[i].on_llm_end(LLMResult(generations=[gen]))
        return LLMResult(generations=generations)
    
    async def _agenerate(self, prompts: list[str], stop: list[str] | None = None, run_managers: list[AsyncCallbackManagerForLLMRun] | None = None, **kwargs) -> LLMResult:
        # Simplified async version for example
        return self._generate(prompts, stop, None, **kwargs) # Non-async managers for simplicity here

    def generate_prompt(self, prompts: list[PromptValue], stop: list[str] | None = None, callbacks: Callbacks = None, **kwargs) -> LLMResult:
        prompt_strings = [str(p) for p in prompts]
        # Simplified manager handling for dummy
        return self._generate(prompt_strings, stop=stop, **kwargs)

    async def agenerate_prompt(self, prompts: list[PromptValue], stop: list[str] | None = None, callbacks: Callbacks = None, **kwargs) -> LLMResult:
        prompt_strings = [str(p) for p in prompts]
        # Simplified manager handling for dummy
        return await self._agenerate(prompt_strings, stop=stop, **kwargs)

    @property
    def _llm_type(self) -> str: return "dummy"


# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 384 # Example dimension for MiniLM

# 1. Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": DEVICE}
)

# 2. Vector Store
vectorstore = Chroma(
    collection_name="nmret_readme_example",
    embedding_function=embedding_model,
    persist_directory="./chroma_db_readme_example",
)
# Add some initial data if needed
vectorstore.add_texts(
    ["Initial context document 1.", "Another piece of information."],
    metadatas=[{"source": "readme", "memory_id": f"readme_{i}"} for i in range(2)],
)


# 3. Titans Neural Memory Wrapper
# Configure NeuralMemory parameters as needed
nm_kwargs = {
    "mem_dim": 64, 
    "layers": 1, 
    "heads": 2, # embedding_dim (384) must be divisible by heads (2)
    "chunk_size": 128,
    "use_accelerated_scan": True, # Set to False if assoc-scan not installed
}
titans_wrapper = TitansNeuralMemoryWrapper(
    embedding_dim=EMBEDDING_DIM, 
    device=DEVICE, 
    neural_memory_kwargs=nm_kwargs
)

# 4. Contextual Memory Wrapper
contextual_memory = VectorStoreContextualMemory(
    vectorstore=vectorstore, 
    embedding_model=embedding_model
)

# 5. LLM
llm = DummyRunnable() # Replace with your actual LLM instance

# 6. LightThinker Compressor
compressor = LightThinkerCompressor(llm=llm, embedding_model=embedding_model)

# 7. Create the Retriever
retriever = NeuralMemoryRetriever(
    vectorstore=vectorstore,
    neural_memory=titans_wrapper,
    contextual_memory=contextual_memory,
    compressor=compressor,
    llm=llm,
    embedding_model=embedding_model,
    device=DEVICE,
    # Configuration
    reasoning_steps=2,        # Number of reasoning loops
    top_k_initial=3,          # K for initial vector store search
    top_k_contextual=2,       # K for contextual memory search per step
    compress_intermediate=True,# Enable LLM thought compression
    update_memory_on_final=False, # Update Titans memory after each step
    update_titans_with="docs_and_llm", # Data source for Titans update 
)

# --- Run a Query ---
query = "What is the main topic discussed?"
print(f"Invoking retriever with query: '{query}'")
results = retriever.invoke(query) 

print("
--- Retrieval Complete ---")
print(f"Final Documents Returned: {len(results)}")
for i, doc in enumerate(results):
    doc_id = doc.metadata.get("memory_id", "N/A")
    content_preview = doc.page_content[:100]
    print(f"  - Doc {i}: ID: {doc_id}, Content: {content_preview}...")

# Example of saving/loading state (if needed)
# state = titans_wrapper.get_state()
# # ... save state ...
# # ... load state ...
# titans_wrapper.load_state(loaded_state)

```

## Dependencies

Key Python packages:

*   `langchain-core`: For core LangChain abstractions.
*   `titans-pytorch`: The neural memory engine. (Ensure it's installed correctly)
*   `torch`: PyTorch library.
*   `numpy`: Numerical operations.
*   `langchain-chroma` / `chromadb`: Example vector store. (Or your chosen `VectorStore` package)
*   `langchain-community` / `sentence-transformers`: Example embeddings. (Or your chosen `Embeddings` package)
*   `langchain-openai` / `langchain-anthropic` / etc.: For the `BaseLanguageModel` used in the compressor and reasoning steps.
*   `uv`: For package management (optional, but used in `pyproject.toml`).
*   `assoc-scan`: Optional, for accelerated `titans-pytorch` operations.

See `pyproject.toml` and `uv.lock` for specific versions.

## License

Uses the Apache 2.0 License as per the `LICENSE` file.
