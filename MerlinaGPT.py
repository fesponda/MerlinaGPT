import openai
import os
import pandas as pd
from llama_index import StorageContext, load_index_from_storage
from llama_index import  GPTVectorStoreIndex, SimpleDirectoryReader,PromptHelper,LLMPredictor
from llama_index import QuestionAnswerPrompt, LangchainEmbedding
from llama_index.node_parser import SimpleNodeParser
from llama_index.indices.service_context import ServiceContext
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import *
import streamlit as st



appOn=True
if appOn:
	INDEXGPTDIR="./"
	DATOSGPTDIR="/Users/fesponda/Morgana/gpt/datos/"
	os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
	#os.environ["OPENAI_API_KEY"] = LLAVE

else:
	INDEXGPTDIR="/Users/fesponda/Morgana/gpt/code/"
	DATOSGPTDIR="/Users/fesponda/Morgana/gpt/datos/"
	os.environ["OPENAI_API_KEY"] = LLAVE

def construct_index(reIndex=False):
    if reIndex:
            #embed_model = LangchainEmbedding(
        #    OpenAIEmbeddings(
        #        query_model_name='text-embedding-ada-002'
        #    )
        #)
        embed_model = LangchainEmbedding(OpenAIEmbeddings(openai_api_key=LLAVE))
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', max_tokens=1000) #era jala con 256 para FAQ
        llm_predictor = LLMPredictor(llm=llm)
        chunk_len=1000# con chunklen muy chico empieza a mezclar respuestas!!!
        chunk_overlap=20
        splitter = TokenTextSplitter(chunk_size=chunk_len, chunk_overlap=chunk_overlap)
        node_parser = SimpleNodeParser(
             text_splitter=splitter, include_extra_info=True, include_prev_next_rel=False
        )
        prompt_helper = PromptHelper.from_llm_predictor(
             llm_predictor=llm_predictor,
        )
        service_context = ServiceContext.from_defaults(
             llm_predictor=llm_predictor,
             prompt_helper=prompt_helper,
             embed_model=embed_model,
             node_parser=node_parser, # con esto mal no encuentra las respuestas en un FAQ
        )
        documents = SimpleDirectoryReader(
            input_dir=DATOSGPTDIR).load_data()
        index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
        index.storage_context.persist(INDEXGPTDIR)
    else:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=INDEXGPTDIR)
        # load index
        index = load_index_from_storage(storage_context)
    return index

def getqueryQA(index):
	QA_PROMPT_TMPL = (
	"Considerando la informacion de contexto a continuacion. \n"
	"---------------------\n"
	"{context_str}"
	"\n---------------------\n"   
	"por favor contesta la siguiente pregunta: {query_str}\n"
	"primero analiza todos los datos y luego elabora la respuesta"
	"resume la respuesta en 50 palabras o menos"
	"contesta en español"
	"Contesta 'no tengo la respuesta a esa pregunta. Por favor contacta a uno de nuestros asesores' si la respuesta no en el contexto"

	)
	QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

	query_engine = index.as_query_engine(
	    text_qa_template=QA_PROMPT
	)
	return query_engine

def qryGPT(pregunta):
	#if 'index' not in st.session_state:
	#	st.session_state['index']=construct_index(reIndex=False)
	#if 'query_engine' not in st.session_state:
	#	st.session_state['query_engine']=getqueryQA(st.session_state['index'])
	response=st.session_state['query_engine'].query(pregunta)
	r=str(response.response).strip().strip('.')
	st.write(f"""{r}""")



if 'index' not in st.session_state:
	st.session_state['index']=construct_index(reIndex=False)
if 'query_engine' not in st.session_state:
	st.session_state['query_engine']=getqueryQA(st.session_state['index'])


st.title("Pericles")
valor=st.text_input("Pregunta", key="pregunta")
prev_qry = ""
if (prev_qry != valor):
	prev_qry = valor
	try:
		qryGPT(st.session_state.pregunta)
	except:
		st.write("intente de nuevo")
#st.button("quibo",on_click=pru)





