import os 
from constant import openapi_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ["OPEN_API_KEY"] = openapi_key

#  Stream lit Framework

st.title("Mini Wiki")
input_text = st.text_input("Hey! What are you looking for ?")


# Prompt template

first_prompt = PromptTemplate(
    
    input_variables = ["input"],
    template = "Brief about the following topic {input}"
    
)




# Open AI LLMS 

llm = OpenAI(openai_api_key = openapi_key, temperature = 0.8)
chain1 = LLMChain(llm = llm, prompt = first_prompt, verbose = True, output_key = 'title')


# Prompt Chaining
second_prompt = PromptTemplate(
    
    input_variables = ["title"],
    template = "  Tell me about the history of {title}"
    
)

chain2 = LLMChain(llm = llm, prompt = first_prompt, verbose = True, output_key = 'DOB')


#  Combining Two chains 
# Using simpleSequentialChain

# parent_chain = SimpleSequentialChain(chains = [chain1, chain2], verbose = True)

# ---> The Problem with simpleSequentialChain is that only the last output is shown . Therefore we will be using sequentialChain


parent_chain = SequentialChain(chains = [chain1, chain2],input_variables =['input'], output_variables=['title', 'DOB'], verbose = True)



if input_text:
    
    # st.write(parent_chain.run(input_text))
    
    st.write(parent_chain({'input': input_text}))
    