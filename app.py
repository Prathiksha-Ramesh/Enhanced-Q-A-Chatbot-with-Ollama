from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

#langsmith tracking

os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']='Q&A Chatbot with OLLAMA'

#prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        ('system','you ate a helpful assistant.Please respond to the user queries '),
        ('user','Question:{question}')
    ]
)
def generate_response(question,engine,temperature,max_tokens):  #temperature-0 (model not be creative and 1-model is much more creative)

    llm=Ollama(model=engine)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer


#title of the app
st.title('Enhanced Q&A chatbot with ollama')
st.sidebar.title('Settings')


#drop down to select various open ai models:
engine=st.sidebar.selectbox('Select an ollama model',['gemma','gemma2','mistral','phi3'])

#adjust response parameter:
temperature=st.sidebar.slider('Temperature',min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider('Max Tokens',min_value=50,max_value=300,value=100)

#main interface for user input:
st.write("Go ahead and ask any question")
user_input=st.text_input('You:')
if user_input:
    response=generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)
else:
    st.write('Please provide the query')

