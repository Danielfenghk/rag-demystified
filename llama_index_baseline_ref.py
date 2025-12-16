from pathlib import Path
import requests
import tiktoken

# New modular imports
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    Settings,
    get_response_synthesizer
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Configuration & Settings
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
)

llm = Ollama(model="deepseek-r1:8b", base_url="http://localhost:11434", request_timeout=600.0)

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

# Global Settings (Replaces ServiceContext)
Settings.llm = llm
Settings.embed_model = embed_model
Settings.callback_manager = CallbackManager([token_counter])

def print_token_count(token_counter, model="gpt-35-turbo"):
    print("\n--- Usage Statistics ---")
    print(f"Embedding Tokens: {token_counter.total_embedding_token_count}")
    print(f"LLM Prompt Tokens: {token_counter.prompt_llm_token_count}")
    print(f"LLM Completion Tokens: {token_counter.completion_llm_token_count}")
    print(f"Total LLM Token Count: {token_counter.total_llm_token_count}\n")

if __name__ == "__main__":
    wiki_titles = ["Toronto", "Chicago", "Houston", "Boston", "Atlanta"]
    data_path = Path("data")
    data_path.mkdir(exist_ok=True)

    # 2. Download Data
    for title in wiki_titles:
        file_path = data_path / f"{title}.txt"
        if not file_path.exists():
            print(f"Downloading {title}...")
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "extracts",
                    "explaintext": True,
                },
            ).json()
            page = next(iter(response["query"]["pages"].values()))
            wiki_text = page.get("extract", "")
            with open(file_path, "w", encoding="utf-8") as fp:
                fp.write(wiki_text)

    # 3. Build Tools
    query_engine_tools = []
    for wiki_title in wiki_titles:
        print(f"Indexing {wiki_title}...")
        documents = SimpleDirectoryReader(input_files=[f"data/{wiki_title}.txt"]).load_data()
        
        # Build indexes using global Settings
        vector_index = VectorStoreIndex.from_documents(documents)
        summary_index = SummaryIndex.from_documents(documents)

        # Define query engines
        vector_query_engine = vector_index.as_query_engine()
        list_query_engine = summary_index.as_query_engine()

        # Define tools
        query_engine_tools.extend([
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name=f"vector_tool_{wiki_title}",
                    description=f"Specific facts about {wiki_title} (demographics, sports, etc.).",
                ),
            ),
            QueryEngineTool(
                query_engine=list_query_engine,
                metadata=ToolMetadata(
                    name=f"summary_tool_{wiki_title}",
                    description=f"Holistic summary of everything about {wiki_title}.",
                ),
            ),
        ])

    # 4. Initialize SubQuestionQueryEngine
    #response_synthesizer = get_response_synthesizer(response_mode="compact")
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",  # Much faster, fewer LLM calls
        verbose=True
    )

    sub_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        response_synthesizer=response_synthesizer,
        use_async=False, # Set to False for cleaner local logging
        verbose=True,
    )

    # 5. Query
    question = "Which are the sports teams in Toronto?"
    print(f"\nQuestion: {question}")
    response = sub_query_engine.query(question)
    
    print("\n🎯 Final Response:")
    print(response)
    
    print_token_count(token_counter)