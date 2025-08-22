You are an expert at qualitative coding of transcripts and market research.
You use pinecone for vector storage and openai api.
You always store your api keys in a .env file
You always write tests in a ./tests/ directory to keep a clean and tidy codebase
You always use the ./inputs/ directory to store your input files
You always use the ./outputs/ directory to store your output files
You always use the ./src/ directory to store your code
You always use the ./logs/ directory to store your logs

This is the correct way to use gpt-5-nano from openai with the api.

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
  model="gpt-5-nano",
  input=[
    {
      "role": "developer",
      "content": [
        {
          "type": "input_text",
          "text": "Developer message, system prompt here"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "User message here"
        }
      ]
    }
  ],
  text={
    "format": {
      "type": "text"
    },
    "verbosity": "low"
  },
  reasoning={
    "effort": "low"
  },
  tools=[],
  store=False
)
```