This tutorial is about LangChains document loader, embedding, and vector store abstractions. These are designed to support the retrieval of data (from vector databased and other sources) for integration with LLM workflows. They are important for applications that fetch data to be reasoned over as part of model inference, as in the case of RAG.

We are going to build a search engine over a PDF document. This will allow us to retrieve passages in the PDF that are similar to an input query.

#### Concepts 
This guide focuses on the retrieval of text data. We will cover the following concepts:
- Documents and document loaders
- Text splitters
- Embeddings
- Vector stores and retrievers

#### Setup

Juypter notebook

`pip install langchain-community pypdf`

Configure LangSmith:
```
import os 
import getpass

os.eniron["LANGSMITH_TRACING"] = "true"
os.enivron["LANGSMITH_API_KEY"] = getpass.getpass("Enter LangSmith API Key: ")
```

#### Documents and Document Loaders
Langchain implements a Document abstraction, which is intended to represent a unit of text and associated metadata. It has three attributes:
- `page_content`: a string representing the content
- `metadata`: a dict containing arbitrary metadata
- `id` (optional) a string identifier for the document

The `metadata` attribute can capture information about the source of the document, its relationship to other documents, and other information. Note that an individual `Document` object often represents a chunk of a larger document.

We can generate sample documents when desired:
```
from langchain_core.documents import Document

documents = [
	Document(
	page_content="Dogs are great companians, known for their loyalty and friendliness."
	metadata={"source": "mammal-pets-doc"},
	),
	Document(
	page_content="Cats are independant pets that often enjoy their own space.",
	metadata={"source": "mammal-pets-doc"},
	),
]
```

LangChains ecosystem implements document loaders that integrate with hundreds of common sources. This makes it easy to incorporate data from these sources into your AI application.

#### Loading documents
Lets load a PDF into a sequence of `Document` objects. There is a sample PDF in the LangChain repo - a 10k filing for Nike from 2023. We can consult the LangChain documentation for available PDF document loaders. We will use PyPDFLoader. More info on PDF document loaders: https://python.langchain.com/docs/how_to/document_loader_pdf/
```
from langchain_community.document_loaders import PyPDFLoader

file_path = "../example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
```

PyPDFLoader loads one `Document` object per PDF page. For each, we can easily access:
- the string content of the page
- Metadata containing the file name and page number
```
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)
```

![[Pasted image 20250120205652.png]]

#### Splitting
For both information retrieval and downstream question-answering purposes, a page may contain too broad of information, or too much. Our end goal is to retrieve `Document` objects that answer an input query, and further splitting our PDF will help ensure that the meanings of relevant portions of the document are not "washed out" by surrounding text. 

We can use text splitters for this purpose. We will use a simple text splitter that partitions based on characters. We will split our documents into chunks of 1000 characters with 200 characters of overlap between chunks. The overlap helps mitigate the possibility of separating a statement from important context related to it. We will use the `RecursiveTextSplitter`, which will recursively split the document using common separators like new lines until each chunk is the appropriate size. (RecursiveTextSplitter is the recommended text splitter for generic use cases.)

We set `add_start_index=True` so the character index where each split Document starts within the initial Document is preserved as metadata attribute `start_index`

Guide for more detail about working with PDFs, such as extracting text from specific sections and images: https://python.langchain.com/docs/how_to/document_loader_pdf/
```
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
len(all_splits)
```

#### Embeddings
Vector search is a common way to store and search over unstructured data (such as unstructured text). The idea is to store numeric vectors that are associated with the text. Given a query, we can embed it as a vector of the same dimension and use vector similarity metrics (such as cosine similarity) to identify related text. 

LangChain supports embeddings from dozens of providers. These models specify how text should be converted into a numeric vector. Lets select a model:
`pip install -qU langchain-openai`
```
import os
import getpass

if not os.environ.get("OPENAI_API_KEY"):
	os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API Key: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```
```
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])
```
We have now created a model that generates text embeddings. Next we will store them in a special data structure that supports efficient similarity search.
![[Pasted image 20250120211126.png]]

#### Vector Stores
LangChain VectorStore objects contain methods for adding text and `Document` objects to the store, and querying them using various similarity metrics. They are often initialized with embedding models, which determine how text data is translated to numeric vectors.

LangChain includes a suite of integrations with different vector store technologies. Some vector stores are hosted by a provider (cloud) and require specific credentials to use; some (such as Postgres) run in separate infrastructure that can be run locally or via a third-party; others can run in-memory for lightweight workloads. We will select a vector store:
`pip install -qU langchain-core`
```
from langchain_core.vector_stores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
```
We can now index the documents:
```
ids = vector_store.add_documents(documents=all_splits)
```
Most vector store implementations will allow you to connect to an existing vector store by providing a client, index name or other information. https://python.langchain.com/docs/integrations/vectorstores/
We can now query the vector store. VectorStore includes methods for querying:
- Synchronously and asynchronously 
- By string query and by vector
- With and without returning similarity scores
- By similarity and maximum marginal relevance (to balance similarity with query to diversity in retrieved results)
The methods will generally include a list of Document objects in their outputs

#### Usage
Embeddings typically represent text as a "dense" vector such that texts with similar meanings are geometrically close. This lets us retrieve relevant information just by passing in a question, without knowledge of any specific key terms used in the document. 

Return documents based on similarity to a string query:
```
results = vector_store.similarity_search(
	"How many distribution centers does Nike have in the US?"
)

print(results[0])
```
![[Pasted image 20250121181444.png]]

Async query:
```
results = await vector_store.asimilarity_search("When was Nike incorporated?")

print(results[0])
```

Return scores:
```
# scores are a distance metric that varies inversely with similarity
# Providers implement different scores; the score here

results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(F"Score: {score}\n")
print(doc)
```

Return documents based on similarity to an embedded query:
```
embedding = embeddings.embed_query("How were Nikes margins impacted in 2023?")

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])
```


#### Retrievers
LangChain `VectorStore` objects do not subclass Runnable. LangChain Retrievers are Runnables, so they implement a standard set of methods (synchronus and asynchronus `invoke` and `batch` operations). Although we can construct retrievers from vector stores, retrievers can interface with non-vector store sources of data, such as external APIs.

We can create a simple version of this ourselves, without subclassing `Retriever`. If we choose what method we wish to use to retrieve documents, we can create a runnable easily. Below we will build one around the `similarity_search` method:
```
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain

@chain
def retriever(query: str) -> List[Document]:
	return vector_store.similarity_search(query, k=1)

retriever.batch(
	[
		"How many distribution centers does Nike have in the US?",
		"When was Nike incorporated?",
	],
)
```

Vectorstores implement an `as_retriever` method that will generate a Retriever, specifically a VectorStoreRetriever. These retrievers include specific `search_type` and `search_kwargs` attributes that identify what methods of the underlying vector store to call, and how to parameratize them. We can replicate the above code with the following:
```
retriever = vector_store.as_retriever(
	search_type="similarity",
	search_kwargs={"k": 1},
)

retriever.batch(
	[
		"How many distribution centers does Nike have in the US?",
		"When was Nike incorporated?",
	],
)
```

![[Pasted image 20250122180657.png]]

`VectorStoreRetriever` supports search types of `similarity` (default), `"mmr"` (maximum marginal relevance) and `similarity_score_threshold`.  We can use the latter to threshold documents output by the retriever by similarity score.

Retrievers can easily be incorporated into more complex applications, such as RAG applications that combine a given question with retrieved context into a prompt for a LLM.

Retrieval strategies can be rich and complex, for ex:
- We can infer hard rules and filters from a query ("using documents published after 2020")
- We can return documents that are linked to the retrieved context in some way
- We can generate multiple embeddings for each unit of context
- We can ensemble results from multiple retrievers
- We can assign weights to documents (e.g. to weigh recent documents higher than older ones)
https://python.langchain.com/docs/how_to/#retrievers

We have just built a semantic search engine over PDF documents!