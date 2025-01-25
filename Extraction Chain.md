We will use tool-calling features of chat models to extract structured information from unstructured text. We will also demonstrate how to use few-shot prompting to improve performance.

Few-shot prompting improves a models performance by feeding it examples of what you want it to do, such as expected inputs and outputs - https://python.langchain.com/docs/concepts/few_shot_prompting/

Jupyter notebook
`pip install --upgrade langchain-core`

Configure LangSmith:
```
import os
import getpass

os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter LangSmith API Key: ")
```

#### The Schema
First, describe what information you want to extract from the text

We will use Pydantic to define an example schema to extract personal information
```
from typing import Optional

from pydantic import BaseModel, Field

class Person(BaseModel):
	"""Information about one person"""

	# ^ Doc-string for the entity Person
	# This doc-string is sent to the LLM as the description of the schema Person
	# It can help to improve extraction results

	# Note that 
	# 1. Each field is an 'optional' - this allows the model to decline                # extracting it
	# 2. Each field has a 'decription' - this description is used by the LLM
	# Having a good description can help improve extraction results
	name: Optional[str] = Field(default=None, description="the name of the person")
	hair_color: Optional[str] = Field(
		default=None, description="The color of a persons hair if known"
	)
	height_in_meters: Optional[str] = Field(
		default=None, description="Height measured in meters"
	)
```

There are two best practices when defining a schema:
1. Document the attributes and the schema itself: This information is sent to the LLM and is used to improve the quality of information extraction
2. Do not force the LLM to make up information! Above we used  `Optional` for the attributes allowing the LLM to output `None` if it doesnt know the answer

Overall: Document the schema well and do not force the LLM to make up answers

#### The Extractor
Create an information extractor using the schema we defined above
```
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagePlaceholder
from pydantic import BaseModel, Field

# Define a custom prompt to provide instructions and any additional context
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take into context into account (e.g,      # include metadata about the document from which the text was extracted)
prompt_template = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are an expert extraction algorithm. "
			"Only extract relevant data from the text. "
			"If you do not know the value of an attribute asked to extract, "
			"return null for the attributes value.",
		),
		# See the how-to about improving performance with refernce examples
		# MessagesPlaceholder('examples'),
		("human", "{text}"),
	]
)
```

Well use a chat model that supports function/tool calling (OpenAI)
`pip install -qU langchain-openai`
```
import os
import getpass 

if not os.environ.get("OPENAI_API_KEY"):
	os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API Key: ")

from langchain-openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
```
`structured_llm = llm.with_structured_output(schema=Person)`

Basic Test:
```
text = "Alan Smith is 6 feet tall and has blonde hair."
prompt = prompt_template.invoke({"text": text})
structured_llm.invoke(prompt)
```
** Since LLMs are generative, it can extract the height in meters even though it was provided in feet

#### Multiple Entities
In most cases, we will be extracting a list of entities rather than a single entity

This can be done using pydantic by nesting models inside one another
```
from typing import List, Optional

from pydantic import BaseModel, Field

class Person(BaseModel):
	"""Information about a person."""

	name: Optional[str] = Field(default=None, description="The name of the person")
	hair_color: Optional[str] = Field(default=None, description="The persons hair color if known")
	height_in_meters: Optional[str] = Field(
	default=None, description="Height measured in meters"
	)

class Data(BaseModel):
	"""Extract data about people."""

	# Creates a model so that we can extract multiple entities.
	people: List[Person]
```

Test:
```
structured_llm = llm.with_structured_output(schema=Data)
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
prompt = prompt_template.invoke({"text": text})
structured_llm.invoke(prompt)
```

#### Reference examples
Below is an example of how we can use few-shot prompting. For chat models, this can take the form of a sequence of pairs and response messages demonstrating desired behaviors. 

For example, we can convey the meaning of a symbol by alternating `user` and `assistant` messages:
```
messages = [
	{"role": "user", "content": "2 ❤️ 2"},
	{"role": "assistant", "content": "4"},
	{"role": "user", "content": "2 ❤️ 3"}
	{"role": "assistant", "content": "5"},
	{"role": "user", }
]
```

Structured output often uses tool calling under-the-hood. This typically involves the generation of AI messages containing tool calls, as well as total messages containing the results of tool calls. What should a sequence of messages look like in this case?

Different chat model providers impose different requirements for valid message sequences. Some will accept a (repeating) message sequence of the form:
- User message
- AI message with tool call
- Tool message with result

Others require a final AI message containing some sort of response. 

LangChain includes a utility function tool_example_to_messages that will generate a valid sequence for most model providers. It simplifies the generation of structured few-shot examples by just requiring Pydantic representations of the corresponding tool calls.

We will try. We can convert pairs of input strings and desired Pydantic objects to a sequence of messages that can be provided to a chat model. Under the hood, LangChain will format the tool calls to each providers required format.
```
from langchain_core.utils.function_calling import tool_example_to_messages

examples = [
	(
		"The ocean is vast and blue. Its more than 20,000 feet deep.",
		Data(people=[]),
	),
	(
		"Fiona traveled far from France to Spain.",
		Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
	),
]

messages = []

for txt, tool_call in examples:
	if tool_call.people:
		# This final message is optional for some providers
		ai_response = "Detected people."
	else:
		ai_response = "Detected no people."
	messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))
```
By inspecting the result, we can see these two example pairs generated eight messages:
```
for message in messages:
	message.pretty_print()
```

We can compare performance with and without these messages. For example, lets pass a message for which we intend no people to be extracted:
```
message_no_extraction = {
	"role": "user",
	"content": "The solar system is large, but earth only has 1 moon.",
}

structured_llm = llm.with_structured_output(schema=Data)
structured_llm.invoke([message_no_extraction])
```

In this example, the model is liable to erroneously generate records of people.

Because our few-shot examples contain examples of "negatives", we encourage the model to behave correctly in this case:
`structured_llm.invoke(messages + [message_no_extraction])`

More info: https://python.langchain.com/docs/how_to/extraction_examples/

