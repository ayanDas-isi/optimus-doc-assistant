"""
author: Elena Lowery

This code sample shows how to invoke Large Language Models (LLMs) deployed in watsonx.ai.
Documentation: # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
You will need to provide your IBM Cloud API key and a watonx.ai project id (any project)
for accessing watsonx.ai
This example shows a Question and Answer use case for a provided document


# Install the wml api your Python env prior to running this example:
# pip install ibm-watson-machine-learning

# Install chroma
# pip install chromadb

# In some envrironments you may need to install chardet
# pip install chardet

IMPORTANT: Be aware of the disk space that will be taken up by documents when they're loaded into
chromadb on your laptop. The size in chroma will likely be the same as .txt file size
"""

# For reading credentials from the .env file
import os
from dotenv import load_dotenv
import argparse
import chromadb

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from chromadb.utils import embedding_functions

# WML python SDK
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

FILE_TYPE_TXT = "txt"
FILE_TYPE_PDF = "pdf"

# Important: hardcoding the API key in Python code is not a best practice. We are using
# this approach for the ease of demo setup. In a production application these variables
# can be stored in an .env or a properties file

# URL of the hosted LLMs is hardcoded because at this time all LLMs share the same endpoint
url = "https://us-south.ml.cloud.ibm.com"

# These global variables will be updated in get_credentials() function
watsonx_project_id = ""
# Replace with your IBM Cloud key
api_key = ""
model=None
collection=[]
# Use the default embeddings function
default_ef = embedding_functions.DefaultEmbeddingFunction()

def get_credentials():

    load_dotenv()
    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)

# The get_model function creates an LLM model object with the specified parameters

def get_model(model_type,max_tokens,min_tokens,decoding,temperature):
    global model
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
        )


def create_embedding(doc_list,collection_name):
    text_list=[]
    for file_path in doc_list:
        typ=file_path.split('.')[-1]
        if typ == FILE_TYPE_TXT:
            loader = TextLoader(file_path,encoding="1252")
            documents = loader.load()
        elif typ == FILE_TYPE_PDF:
            loader = PyPDFLoader(file_path)
            documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        text_list.extend(texts)

    print(type(texts))

    # Load chunks into chromadb
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name,embedding_function=default_ef)
    collection.upsert(
        documents=[doc.page_content for doc in texts],
        ids=[str(i) for i in range(len(texts))],  # unique for each doc
    )

    return collection


def get_answer(question):
    relevant_chunks = collection.query(
        query_texts=[question],
        n_results=3,
    )

    context = "\n\n\n".join(relevant_chunks["documents"][0])
    # Please note that this is a generic format. You can change this format to be specific to llama
    prompt = (f"{context}\n\nPlease answer a question using this "
            + f"text. "
            + f"If the question is unanswerable, say \"unanswerable\"."
            + f"{question}")

        # Let's review the prompt
    print("----------------------------------------------------------------------------------------------------")
    #print("*** Prompt:" + prompt + "***")
    print("----------------------------------------------------------------------------------------------------")

    generated_response = model.generate(prompt=prompt)
    response_text = generated_response['results'][0]['generated_text']
    return response_text

def create_prompt(doc_list, collection_name):
    global collection
    # Create embeddings for the text file
    collection = create_embedding(doc_list,collection_name)



#python .\Scripts\FRP_helper.py --docs "['RFP for Data Lake Solution BOI_ Final dt 30.06.2022_Last_final2022-07-01 110547185 (1).pdf','SynCanSoW.pdf']"
def set_up_doc(file_path):

    print(file_path)
    doc_list=[file_path]
    # Get the API key and project id and update global variables
    get_credentials()

    # question = "What did the president say about jobs?"
    # question = "What did the president say about inflation?"
    # Provide the path relative to the dir in which the script is running
    # In this example the .txt file is in the same directory
    # In this example the .pdf file is in the same directory
    # You may also have to hard-code the path if you cannot get the relative path to work
    # file_path = "C:/Users/xxxxxxxxx/Documents/VS Code/state_of_the_union.txt"

    collection_name = "state_of_the_union"
    
    # Test answering questions based on the provided .txt file
    answer_questions_from_doc(api_key,watsonx_project_id,doc_list,collection_name)

def answer_questions_from_doc(request_api_key, request_project_id, doc_list,collection_name):

    # Update the global variable
    globals()["api_key"] = request_api_key
    globals()["watsonx_project_id"] = request_project_id

    # Specify model parameters
    model_type = ModelTypes.GRANITE_13B_CHAT_V2
    max_tokens = 300
    min_tokens = 20
    decoding = DecodingMethods.GREEDY
    temperature = 0.95

    get_model(model_type, max_tokens, min_tokens, decoding, temperature)

    # Get the prompt
    complete_prompt = create_prompt(doc_list, collection_name)


    #return response_text

