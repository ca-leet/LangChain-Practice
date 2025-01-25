Tagging means labeling a document with classes such as:
- Sentiment
- Language
- Style (formal, informal, etc.)
- Covered topics
- Political tendency
![[Pasted image 20250122193611.png]]

#### Overview
Tagging has a few components:
- `function`: like extraction, tagging uses functions to specify how the model should tag a document
- `schema`: defines how we want to tag the document

#### Quickstart
Here is a very straightforward example of how we can use OpenAIs tool calling for tagging in LangChain - we will use `with_structured_output` method supported by OpenAI models.
`pip install --upgrade --quiet langchain-core`

Select a chat model (we will use OpenAI):
`pip install langchain-openai`
```
import os
import getpass

if not os.environ.get["OPENAI_API_KEY"]:
	os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key: ")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
```

Next, specify a Pydantic model with a few properties and their expected type in our schema:
```
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
	"""
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
	"""
)

class Classification(BaseModel):
	sentiment: str = Field(description="The sentiment of the text")
	aggressiveness: int = Field(
		description="How aggressive the text is on a scale of 1 to 10"
	)
	language: str = Field(description="The language the text is written in")

# LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
	Classification
)
```

```
inp = "Estoy incredeiblemente contento de haberte conocido! Creo que muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

response
```

If we wanted a dictionary output, we could call `.dict()` - EDIT - Use `model_dump` instead (`response.model_dump()`)
```
inp = "Estoy incredeiblemente contento de haberte conocido! Creo que muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

response.dict()
```

#### Finer Control
Careful schema definition gives us more control over the models output

Specifically, we can define:
- Possible values for each property
- Description to make sure that the model understands the property
- Required properties to be returned

We will redeclare our Pydantic model to control for each of the previously mentioned aspects using enums:
```
class Classification(BaseModel):
	sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
	aggressiveness: int = Field(
		...,
		description="describes how aggressive the statement is, the higher the number the more aggressive",
		enum=[1, 2, 3, 4, 5],
	)
	language: str = Field(
		..., enum=["spanish", "english", "french", "german", "italian"]
	)
```
```
tagging_prompt = ChatPromptTemplate.from_template(
	"""
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
	Classification
)
```
Now the answers should be provided in a way we expect
```
inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
prompt = tagging_prompt.invoke({"input": inp})
llm.invoke(prompt)
```
```
inp = "The weather is ok here, I can go outside without much more than a coat"
prompt = tagging_prompt.invoke({"input": inp})
llm.invoke(prompt)
```
