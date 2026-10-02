import os

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import uvicorn


# Load environment variables
load_dotenv(find_dotenv())

groq_api_key = os.environ["GROQ_API_KEY"]


# LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# Output parser
parser = StrOutputParser()


# Prompt
system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])


# Chain
chain = prompt_template | llm | parser


# FastAPI app
app = FastAPI(
    title="simpleTranslator",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)


# Add LangServe routes
add_routes(
    app,
    chain,
    path="/chain",
)


# Run server
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port=8000
    )