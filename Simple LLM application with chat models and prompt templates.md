LangChain tutorial

- this application will translate text from English to another language
- this is a simple LLM call plus some prompting

High Level:
- using language models
- using prompt templates
- Debugging and tracing application with LangSmith

Jupyter Notebook

#### Installation
`pip install langchain`

**LangSmith:**
Applications with multiple steps and multiple invocations of LLM calls can get complex - it is crucial to be able to inspect what exactly is going on inside your chain or agent - LangSmith
```
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API Key: ")
```

#### Using Language Models
First we will learn to use the language model by itself. LangChain supports many different language models you can use interchangeably. We will use OpenAI
`pip install -qU langchain-openai`
```
import os
import getpass

if not os.environ.get("OPENAI_API_KEY"):
	os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key: ")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
```

First we will use the model directly. ChatModels are instances of LangChain runnables, which means they expose a standard interface for interacting with them. to simply call the model, we can pass in a list of messages to the .invoke method
```
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
	SystemMessage("Translate the following from English to Italian"),
	HumanMessage("hi")
]

model.invoke(messages)
```

** At this point if LangSmith is enabled, we can see this run is logged and see the LangSmith trace. The trace reports token usage information, latency, standard model parameters, and other information

Note that ChatModels receive message objects as input and generate message objects as output. In addition to text content, message objects convey conversational roles and hold important data, such as tool calls and token usage counts.

LangChain also supports chat model inputs via strings or OpenAI format. The following are equivalent:
```
model.invoke("Hello")

model.invoke([{"role": "user", "content": "Hello"}])

model.invoke([HumanMessage("Hello")])
```


#### Streaming
Because chat models are runnables, they expose a standard interface that includes async and streaming modes of invocation. This allows us to stream individual tokens from a chat model:
```
for token in model.stream(messages):
	print(token.content, end="|")
```


#### Prompt Templates
Right now we are passing a list of messages directly into the language model. Where do these messages come from? Usually, its constructed from a combination of user input and application logic. This application logic usually takes the raw user input and transforms it into a list of messages ready to pass to the language model. Common transformations include adding a system message or formatting a template with the user input.

Prompt templates are a concept in LangChain designed to assist with this transformation. They take raw user input and return data (a prompt) that is ready to pass into the language model.

Lets create a prompt template here. It will take two user variables:
- `language`: the language to translate the text into
- `text`: the text to translate
```
from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
	[("system", system_template), ("user", "{text}")]
)
```
Note that `ChatPromptTemplate` supports multiple message roles in a single template. We format the `language` parameter into the system message, and the user `text` into a user message.

The input to this prompt template is a dictionary. We can play around this this prompt template by itself to see what it does by itself
```
prompt = prompt_template.invoke({"language": "Italian", "text": "Hi!"})

prompt
```
We can see that it returns a `ChatPromptValue` that consists of two messages. If we want to access the messages directly we do:
`prompt.to_messages()`

Then, we can invoke the chat model on the formatted prompt:
```
response = model.invoke(prompt)
print(response.content)
```

#### Conclusion
What did we accomplish?
- created my first LLM application
	- interacted with an LLM via OpenAI API
	- created a prompt template
	- used LangSmith to view statistics of application


#### Runnable Interface
The Runnable Interface is the foundation for working with LangChain components, and its implemented across many of them, such as language models, output parsers, retrievers, compiled LangGraph graphs and more

**It allows developers to interact with various LangChain components in a consistent and predictable manner**

The Runnable way defines a standard interface that allows a Runnable component to be:
- **Invoked**: A single input is transformed into an output
- **Batched**: Multiple inputs are efficiently transformed into an output
- **Streamed**: Outputs are streamed as they are produced
- **Inspected**: Schematic information about Runnables input, output, and configuration can be accessed
- **Composed**: Multiple Runnables can be composed to work together using the LangChain Expression Language (LCEL) to create complex pipelines

LCEL cheat sheet: https://python.langchain.com/docs/how_to/lcel_cheatsheet/

#### Optimized Parallel Execution (batch)
LangChain runnables offer a built-in `batch` (and `batch_as_completed`) API that allows you to process multiple inputs in parallel. 

Using these methods can significantly improve performance when needing to process multiple independent inputs, as processing can be done in parallel instead of sequentially

The two batching options are:
- `batch`: Process multiple inputs in parallel, returning results in the same order as inputs
- `batch_as_completed`: Process multiple inputs as parallel, returning results as they complete. Results may arrive out of order, but each includes the input index for matching

The default implementation of `batch` and `batch_as_completed` use a thread pool executor to run the `invoke` method in parallel. This allows for efficient parallel execution without the need for users to manage threads, and speeds up code that is I/O


