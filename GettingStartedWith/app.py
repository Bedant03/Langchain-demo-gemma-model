import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

##langsmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

#prompt template 
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Please respond to the question asked "),
        ("user","question:{question}")
    ]
)


#streamlit framework
st.title("langchain Demo With Google gemma model")
input_text=st.text_input("What question u have in mind?")


#Ollama model gemma:2b
llm=Ollama(model="gemma:2b")
outputparser=StrOutputParser()
chain=prompt|llm|outputparser


if input_text:
    st.write(chain.invoke({"question":input_text}))