from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
os.environ["LANGCHAIN_PROJECT"]= "Sequential_Model_App"

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="conversational",
    temperature= 0.5
)
model= ChatHuggingFace(llm= llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config= {'tags': ['Ilm app', 'report generation', 'summarization'],
         'metadata': {'model1': 'Huggingface ', "model_temp": 0.5, 'parser': 'sting_output_parser'}
         }

result = chain.invoke({'topic': 'Unemployment in India'}, config= config)

print(result)
