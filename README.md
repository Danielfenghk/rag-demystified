# Demystifying Advanced RAG Pipelines

Retrieval-Augmented Generation (RAG) pipelines powered by large language models (LLMs) are gaining popularity for building end-to-end question answering systems. Frameworks such as [LlamaIndex](https://github.com/run-llama/llama_index) and [Haystack](https://github.com/deepset-ai/haystack) have made significant progress in making RAG pipelines easy to use. While these frameworks provide excellent abstractions for building advanced RAG pipelines, they do so at the cost of transparency. From a user perspective, it's not readily apparent what's going on under the hood, particularly when errors or inconsistencies arise. 

In this [EvaDB](https://github.com/georgia-tech-db/evadb) application, we'll shed light on the inner workings of advanced RAG pipelines by examining the mechanics, limitations, and costs that often remain opaque.

<p align="center">
  <img width="70%" src="images/intro.png" title="llama working on a laptop to retrieve data" >
  <br>
  <b><i>Llama working on a laptop</i> 🙂</b>
</p>

## Quick start

If you want to jump right in, use the following commands to run the application:

```
pip install -r requirements.txt

echo OPENAI_API_KEY='yourkey' > .env
python complex_qa.py
```

## RAG Overview

Retrieval-augmented generation (RAG) is a cutting-edge AI paradigm for LLM-based question answering.
A RAG pipeline typically contains:

1. **Data Warehouse** - A collection of data sources (e.g., documents, tables etc.) that contain information relevant to the question answering task.

2. **Vector Retrieval** - Given a question, find the top K most similar data chunks to the question. This is done using a vector store (e.g., [Faiss](https://faiss.ai/index.html)).

3. **Response Generation** - Given the top K most similar data chunks, generate a response using a large language model (e.g. GPT-4).

RAG provides two key advantages over traditional LLM-based question answering:
1. **Up-to-date information** - The data warehouse can be updated in real-time, so the information is always up-to-date.

2. **Source tracking** - RAG provides clear traceability, enabling users to identify the sources of information, which is crucial for accuracy verification and mitigating LLM hallucinations.

## Building advanced RAG Pipelines

To enable answering more complex questions, recent AI frameworks like LlamaIndex have introduced more advanced abstractions such as the [Sub-question Query Engine](https://gpt-index.readthedocs.io/en/latest/examples/query_engine/sub_question_query_engine.html).

In this application, we'll demystify sophisticated RAG pipelines by using the Sub-question Query Engine as an example. We'll examine the inner workings of the Sub-question Query Engine and simplify the abstractions to their core components. We'll also identify some challenges associated with advanced RAG pipelines.

### The setup

A data warehouse is a collection of data sources (e.g., documents, tables etc.) that contain information relevant to the question answering task.

In this example, we'll use a simple data warehouse containing multiple Wikipedia articles for different popular cities, inspired by LlamaIndex's [illustrative use-case](https://docs.llamaindex.ai/en/stable/examples/index_structs/doc_summary/DocSummary.html). Each city's wiki is a separate data source. Note that for simplicity, we limit each document's size to fit within the LLM context limit.

Our goal is to build a system that can answer questions like:
1. *"What is the population of Chicago?"*
2. *"Give me a summary of the positive aspects of Atlanta."*
3. *"Which city has the highest population?"*

As you can see, the questions can be simple factoid/summarization questions over a single data source (Q1/Q2) or complex factoid/summarization questions over multiple data sources (Q3).

We have the following *retrieval methods* at our disposal:

1. **vector retrieval** - Given a question and a data source, generate an LLM response using the top-K most similar data chunks to the question from the data source as the context. We use the off-the-shelf FAISS vector index from [EvaDB](https://github.com/georgia-tech-db/evadb) for vector retrieval. However, the concepts are applicable to any vector index.

2. **summary retrieval** - Given a summary question and a data source, generate an LLM response using the entire data source as context.

### The secret sauce

Our key insight is that each component in an advanced RAG pipeline is powered by a single LLM call. The entire pipeline is a series of LLM calls with carefully crafted prompt templates. These prompt templates are the secret sauce that enable advanced RAG pipelines to perform complex tasks.

In fact, any advanced RAG pipeline can be broken down into a series of individual LLM calls that follow a universal input pattern:

![equation](images/equation.png)

<!-- LLM input = **Prompt Template** + **Context** + **Question** -->
where:
- **Prompt Template** - A curated prompt template for the specific task (e.g., sub-question generation, summarization)
- **Context** - The context to use to perform the task (e.g. top-K most similar data chunks)
- **Question** - The question to answer

Now, we illustrate this principle by examining the inner workings of the Sub-question Query Engine.

The Sub-question Query Engine has to perform three tasks:
1. **Sub-question generation** - Given a complex question, break it down into a set of sub-questions, while identifying the appropriate data source and retrieval function for each sub-question.
2. **Vector/Summary Retrieval** - For each sub-question, use the chosen retrieval function over the corresponding data source to retrieve the relevant information.
3. **Response Aggregation** - Aggregate the responses from the sub-questions into a final response.

Let's examine each task in detail.

### Task 1: Sub-question Generation

Our goal is to break down a complex question into a set of sub-questions, while identifying the appropriate data source and retrieval function for each sub-question. For example, the question *"Which city has the highest population?"* is broken down into five sub-questions, one for each city, of the form *"What is the population of {city}?".* The data source for each sub-question has to be the corresponding city's wiki, and the retrieval function has to be vector retrieval.

At first glance, this seems like a daunting task. Specifically, we need to answer the following questions:
1. **How do we know which sub-questions to generate?**
2. **How do we know which data source to use for each sub-question?**
3. **How do we know which retrieval function to use for each sub-question?**

Remarkably, the answer to all three questions is the same - a single LLM call! The entire sub-question query engine is powered by a single LLM call with a carefully crafted prompt template. Let's call this template the **Sub-question Prompt Template**.

```
-- Sub-question Prompt Template --

"""
    You are an AI assistant that specializes in breaking down complex questions into simpler, manageable sub-questions.
    When presented with a complex user question, your role is to generate a list of sub-questions that, when answered, will comprehensively address the original question.
    You have at your disposal a pre-defined set of functions and data sources to utilize in answering each sub-question.
    If a user question is straightforward, your task is to return the original question, identifying the appropriate function and data source to use for its solution.
    Please remember that you are limited to the provided functions and data sources, and that each sub-question should be a full question that can be answered using a single function and a single data source.
"""
```

The context for the LLM call is the names of the data sources and the functions available to the system. The question is the user question. The LLM outputs a list of sub-questions, each with a function and a data source.

![task_1_table](images/task_1_table.png)

For the three example questions, the LLM returns the following output:

<details>
  <summary>
    LLM output Table
  </summary>
<table>
<thead>
  <tr>
    <th>Question</th>
    <th>Subquestions</th>
    <th>Retrieval method</th>
    <th>Data Source</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>"What is the population of Chicago?"</td>
    <td>"What is the population of Chicago?"</td>
    <td>vector retrieval</td>
    <td>Chicago</td>
    </tr>
    <tr>
    <td>"Give me a summary of the positive aspects of Atlanta."</td>
    <td>"Give me a summary of the positive aspects of Atlanta."</td>
    <td>summary retrieval</td>
    <td>Atlanta</td>
    </tr>
    <tr>
    <td rowspan=5>"Which city has the highest population?"</td>
    <td>"What is the population of Toronto?"</td>
    <td>vector retrieval</td>
    <td>Toronto</td>
    </tr>
    <tr>
    <td>"What is the population of Chicago?"</td>
    <td>vector retrieval</td>
    <td>Chicago</td>
    </tr>
    <tr>
    <td>"What is the population of Houston?"</td>
    <td>vector retrieval</td>
    <td>Houston</td>
    </tr>
    <tr>
    <td>"What is the population of Boston?"</td>
    <td>vector retrieval</td>
    <td>Boston</td>
    </tr>
    <tr>
    <td>"What is the population of Atlanta?"</td>
    <td>vector retrieval</td>
    <td>Atlanta</td>
    </tr>
</tbody>
</table>
</details>

### Task 2: Vector/Summary Retrieval

For each sub-question, we use the chosen retrieval function over the corresponding data source to retrieve the relevant information. For example, for the sub-question *"What is the population of Chicago?"*, we use vector retrieval over the Chicago data source. Similarly, for the sub-question *"Give me a summary of the positive aspects of Atlanta."*, we use summary retrieval over the Atlanta data source.

For both retrieval methods, we use the same LLM prompt template. In fact, we find that the popular **RAG Prompt** from [LangchainHub](https://smith.langchain.com/hub) works great out-of-the-box for this step.

```
-- RAG Prompt Template --

"""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
```

Both the retrieval methods only differ in the context used for the LLM call. For vector retrieval, we use the top K most similar data chunks to the sub-question as context. For summary retrieval, we use the entire data source as context.

![task_2_table](images/task_2_table.png)

### Task 3: Response Aggregation

This is the final step that aggregates the responses from the sub-questions into a final response. For example, for the question *"Which city has the highest population?"*, the sub-questions retrieve the population of each city and then response aggregation finds and returns the city with the highest population.
The **RAG Prompt** works great for this step as well.

The context for the LLM call is the list of responses from the sub-questions. The question is the original user question and the LLM outputs a final response.

![task_3_table](images/task_3_table.png)

### Putting it all together

After unraveling the layers of abstraction, we uncovered the secret ingredient powering the sub-question query engine - 4 types of LLM calls each with different prompt template, context, and a question. This fits the universal input pattern that we identified earlier perfectly, and is a far cry from the complex abstractions that we started with.
To summarize:
![equation](images/equation.png)
![call_types_table](images/call_types_table.png)

To see the full pipeline in action, run the following commands:

```
pip install -r requirements.txt

echo OPENAI_API_KEY='yourkey' > .env
python complex_qa.py
```

Here is an example of the system answering the question *"Which city with the highest population?"*.

![full_pipeline](images/simple_rag.png)

## Challenges

Now that we've demystified the inner workings of advanced RAG pipelines, let's examine the challenges associated with them.

1. **Question sensitivity** - The biggest challenge that we observed with these systems is the question sensitivity. The LLMs are extremely sensitive to the user question, and the pipeline fails unexpectedly for several user questions. Here are a few example failure cases that we encountered:
    - **Incorrect sub-questions** - The LLM sometimes generates incorrect sub-questions. For example, *"Which city has the highest number of tech companies?"* is broken down into *"What are the tech companies in each city?"* 5 times (once for each city) instead of *"What is the number of tech companies in Toronto?"*, *"What is the number of tech companies in Chicago?"*, etc.
    - **Incorrect retrieval function** - *"Summarize the positive aspects of Atlanta and Toronto."* results in using the vector retrieval function instead of the summary retrieval method.

We had to put in significant effort into prompt engineering to get the pipeline to work for each question. This is a significant challenge for building robust systems.

To verify this behavior, we [implemented the example](llama_index_baseline.py) using the LlamaIndex Sub-question query engine. Consistent with our observations, the system often generates the wrong sub-questions and also uses the wrong retrieval function for the sub-questions, as shown below.

![llama_index_baseline](images/baseline.png)


2. **Cost** - The second challenge is the cost dynamics of advanced RAG pipelines. The issue is two-fold:
    - **Cost sensitivity** - The final cost of the question is dependent on the number of sub-questions generated, the retrieval function used, and the number of data sources queried. Since the LLMs are sensitive to the prompt, the cost of the question can vary significantly depending on the question and the LLM output. For example, the incorrect model choice in the LlamaIndex baseline example above (`summary_tool`) results in a 3x higher cost compared to the `vector_tool` while also generating an incorrect response.
    - **Cost estimation** - Advanced abstractions in RAG frameworks obscure the estimated cost of the question. Setting up a cost monitoring system is challenging since the cost of the question is dependent on the LLM output.


## Conclusion

Advanced RAG pipelines powered by LLMs have revolutionized question-answering systems.
However, as we have seen, these pipelines are not turnkey solutions. Under the hood, they rely on carefully engineered prompt templates and multiple chained LLM calls. As illustrated in this [EvaDB](https://github.com/georgia-tech-db/evadb) application, these pipelines can be question-sensitive, brittle, and opaque in their cost dynamics. Understanding these intricacies is key to leveraging their full potential and paving the way for more robust and efficient systems in the future.


<!-- ## Appendix


To reliably generate the correct format of functions and data sources, we use the powerful [OpenAI function calling](https://openai.com/blog/function-calling-and-other-api-updates) feature paired with Pydantic models. We also use the [Instructor](https://github.com/jxnl/instructor) library to easily generate LLM-ready function schemas.

More details on the full schema definition can be found [here](subquestion_generator.py).

For example, the function schema to choose vector/summary retrieval is as simple as:

```python
class FunctionEnum(str, Enum):
    """The function to use to answer the questions.
    Use vector_retrieval for factoid questions.
    Use summary_retrieval for summarization questions.
    """
    VECTOR_RETRIEVAL = "vector_retrieval"
    SUMMARY_RETRIEVAL = "summary_retrieval"
```

The data source schema definition is also straightforward:
```python
class DataSourceEnum(str, Enum):
    """The data source to use to answer the corresponding subquestion"""
    TORONTO = "Toronto"
    CHICAGO = "Chicago"
    HOUSTON = "Houston"
    BOSTON = "Boston"
    ATLANTA = "Atlanta"
```

All of this can be packaged into a simple Pydantic model:

```python
class QuestionBundle(BaseModel):
    question: str = Field(None, description="The subquestion extracted from the user's question")
    function: FunctionEnum
    data_source: DataSourceEnum
```

Using the Instructor library, we can provide the above schema as the desired output format to OpenAI.
```python
from instructor import OpenAISchema

class SubQuestionBundleList(OpenAISchema):
    subquestion_bundle_list: List[QuestionBundle] = Field(None, description="A list of subquestions - each item in the list contains a question, a function, and a data source")

response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        functions=[QuestionBundle.OpenAISchema],
        ...
)
``` -->

### 不要再強行使用 EvaDB 了
#EvaDB 在 Windows 上的相容性本來就差，加上版本落後、pydantic 衝突、UDF 檔案不存在等問題，會讓你一直卡在各種錯誤中。
#最佳且最乾淨的解決方案：完全移除 EvaDB，改用純 Python + FAISS + sentence-transformers
這是 2025 年絕大多數人跑本地 RAG 的標準做法，穩定、快速、完全相容 Windows，且不會破壞環境。
pip install faiss-cpu sentence-transformers

#EvaDB 的儲存引擎在處理多媒體/文件時，會嘗試建立 symbolic link 來避免複製檔案（節省空間）。
在 Windows 上，建立 symlink 需要特殊權限（SeCreateSymbolicLinkPrivilege）

###  當 deepseek-r1:8b 在 Ollama 上執行 function calling 時，它經常不嚴格遵守 JSON 格式，而是直接回普通文字答案
Llama3.1 和 Qwen2.5 在 Ollama 上 function calling 幾乎 100% 成功
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
### 推薦最終選擇
直接換成 llama3.1:8b — 這是 2025 年底最平衡的選擇：

8B 參數，
CPU 可輕鬆跑
function calling 極其穩定
推理能力接近 deepseek-r1:8b
Meta 官方支援，
Ollama 優化最好

執行：
Bash
ollama pull llama3.1:8b

###最新修改推送到你的 GitHub fork 的完整命令（已驗證可正常運作）：
Bash# 1. 確認目前 remote 正確（應該指向你的 fork）
git remote -v

# 如果輸出不是你的 fork，執行這行修正（只需一次）
git remote set-url origin https://github.com/Danielfenghk/rag-demystified.git

# 2. 添加所有修改檔案
git add .

# 3. 提交修改（寫一個清楚的 commit 訊息）
git commit -m "Refactor for full Ollama support: remove EvaDB, use pure FAISS + sentence-transformers, robust subquestion generation"

# 4. 推送到你的 GitHub fork 的 main 分支
git push origin main


# 1. What are FAISS and Sentence-Transformers?
FAISS (Facebook AI Similarity Search) is a library specifically built for efficient similarity search and clustering of dense vectors. Its core strength is speed and scalability when you need to find the closest vectors (e.g., the most similar text or images) in a massive dataset (billions of vectors). It uses advanced indexing techniques like IVF (Inverted File Index) and HNSW (Hierarchical Navigable Small World) to perform searches that are much faster than a brute-force comparison.

Sentence-Transformers is a Python framework built on top of PyTorch and Hugging Face Transformers. Its purpose is to easily generate high-quality embeddings (dense vector representations) for sentences, paragraphs, and short documents. It provides pre-trained models (e.g., all-MiniLM-L6-v2) that convert text into a numerical vector (e.g., 384 dimensions) where semantically similar texts have vectors that are close together in the vector space.

# 2. What is EvaDB?
EvaDB is a complete AI-centric database system. It's not just a vector search library; it's designed to bring AI/ML models directly into a database workflow. Think of it as PostgreSQL with superpowers for AI. Its key features are:

## SQL Interface for AI: You write SQL queries that call AI functions (e.g., SELECT ChatGPT(query, column) FROM table).

## Built-in Model Management: It can cache, optimize, and run various AI models (for vision, NLP, etc.).

## Vector Search as a Feature: It has built-in support for vector indexing (often using FAISS or similar backends internally) and similarity search, but this is one of many capabilities.

## Data Orchestration: It connects to multiple data sources (PostgreSQL, SQLite, CSVs, etc.) and manages the pipeline from raw data to AI inference.

Why FAISS + Sentence-Transformers Can Feel Like a Replacement for EvaDB
The combination is often positioned as an alternative specifically for building a vector search application for text. Here’s the logic:

Aspect	FAISS + Sentence-Transformers	EvaDB
Primary Goal	Specialized, high-performance vector search for text. A focused pipeline: text -> vector -> store -> search.	General-purpose AI database. Integrates AI models (of all types) directly into data management via SQL.
Flexibility	Modular & DIY. You choose the embedding model, the index type, and manage the pipeline (chunking, ingestion, retrieval) yourself. High control for engineers.	Integrated & Declarative. You describe what you want (in SQL), and EvaDB handles the how. Less boilerplate code.
Strength	Raw speed and scale for search. FAISS is industry-standard for maximum performance on pure vector search. You can fine-tune every parameter.	Developer productivity and simplicity for multi-model AI apps. Easy to chain multiple AI functions, filter by metadata, and connect to existing databases.
Use Case Fit	Best when your core need is semantic search/retrieval at scale (e.g., building a RAG system's retrieval engine, a large-scale recommendation system).	Best when you need to quickly query data with various AI models (e.g., "find videos where a car appears and summarize the transcript," combining object detection and NLP).
The Analogy:

FAISS + Sentence-Transformers is like buying high-performance car parts (engine, transmission, suspension) and building a race car yourself. It's powerful and optimized for the track (search), but you have to assemble and maintain it.

EvaDB is like buying a high-end, feature-rich SUV that has a great built-in navigation system, towing capacity, and a great sound system. It's very capable for many tasks (including off-road/on-road) and more convenient, but it might not be the absolute fastest on a racetrack.

When You Might Choose One Over the Other
Choose FAISS + Sentence-Transformers if:

Vector search is your dominant, performance-critical need. You need the absolute lowest latency and highest recall for semantic search.

You already have a dedicated engineering team comfortable managing the entire pipeline (data loading, embedding generation, index tuning, serving, and updates).

Your application is primarily about retrieving relevant text/documents (like in RAG systems), and you don't need complex SQL-based AI chaining.

Choose EvaDB if:

You want to rapidly prototype or build an application that uses multiple AI functions (e.g., sentiment analysis + question answering + image classification) without writing extensive glue code.

Your team is proficient in SQL and prefers a declarative approach to AI queries.

Your use case involves complex filtering (e.g., "find documents from last week that are similar to this query and have a positive sentiment").

You want a unified system to manage and cache AI models and connect directly to your existing operational databases.

Conclusion
FAISS + Sentence-Transformers doesn't fully "replace" EvaDB. Instead, it provides a more specialized, high-performance alternative for the vector search component that EvaDB also offers.

If your project is a text-centric vector search engine, the DIY approach with FAISS + Sentence-Transformers is a classic, powerful, and scalable choice. If your project is a broader AI-powered data application where vector search is just one part of a larger workflow involving multiple models and data sources, EvaDB's integrated approach can save significant development time and complexity.

In essence: They are tools for different layers of the stack. FAISS is an algorithmic library for a specific task, while EvaDB is a system for orchestrating multiple AI tasks.
