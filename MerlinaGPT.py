import openai
import os
import pandas as pd
from scipy.spatial.distance import cosine

import json
import time
import streamlit as st
from tenacity import retry, wait_random_exponential, stop_after_attempt


appOn=True

if appOn:
	INDEXGPTDIR="./"
	DATOSGPTDIR="/Users/fesponda/Morgana/gpt/datos/"
	os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
	#os.environ["OPENAI_API_KEY"] = LLAVE

else:
    INDEXGPTDIR="/Users/fesponda/Morgana/gpt/code/"
    DATOSGPTDIR="/Users/fesponda/Morgana/gpt/datos/"
    openai.api_key = LLAVE


def get_completion(prompt, model="gpt-3.5-turbo"): # Andrew mentioned that the prompt/ completion paradigm is preferable for this class
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def getChunks(doc_store,query='',mix=False,numChunks=2,forward=0,back=0):
    #doc_store has all the data and embeddings, mix is whether to allow chunks from diff documentos, forward and back to includ adyacent chuncks
    res=[]
    allDistances=[]
    for doc in doc_store:
        alld,r=findclosest(doc_store,doc,query)
        res+=[r] # the top from each document
        allDistances+=alld #all distances
    sList=sorted(allDistances,key=lambda x:x[1])
    #retrieve the closest numChunks from same document
    topDoc=sList[0][2]
    context=""
    chunks=0
    for i,x in enumerate(sList):
        if x[2]==topDoc and not mix:
            context+=doc_store[topDoc][x[0]]['text']
            chunks+=1
        elif mix:
            context+=doc_store[x[2]][x[0]]['text']
            chunks+=1
        if chunks>=numChunks:
            break
    return context
        
def findclosest(documents,name,d1):
    documents=documents[name]
    mindist=1000
    resDoc=d1
    allDistances=[]
    for dockey in documents:
        if d1 != documents[dockey]['vector']:
            distance=cosine(documents[dockey]['vector'],d1)
            allDistances+=[[dockey,distance,name]]
            #print(distance)
            if distance<mindist:
                mindist=distance
                resDoc=dockey
    return allDistances,[resDoc,mindist,name]

def embedding_from_string(input: str, model: str) -> list:
    response = openai.Embedding.create(input=input, model=model)
    embedding = response["data"][0]["embedding"]
    return embedding


def createIndex(reIndex=False):
    if reIndex:
        doc_store={}
        onlyfiles = [f for f in listdir(DATOSGPTDIR) if isfile(join(DATOSGPTDIR, f)) and '.DS' not in f and ('.csv' in f or '.txt' in f) ]
        for dataFile in onlyfiles:
            print(dataFile)
            doc_store=embedDocument(DATOSGPTDIR+dataFile,doc_store,lines=20,model="text-embedding-ada-002")
        
        # Serializing json
        json_object = json.dumps(doc_store, indent=4)

        # Writing to sample.json
        with open(INDEXGPTDIR+"doc_store.json", "w") as outfile:
            outfile.write(json_object)
    else:
        with open(INDEXGPTDIR+'doc_store.json') as json_file:
            doc_store = json.load(json_file)
    return doc_store



from tenacity import retry, wait_random_exponential, stop_after_attempt
def qryGPT(query_text):

    pru=embedding_from_string(query_text,"text-embedding-ada-002")

    context_text=getChunks(st.session_state['index'],pru,mix=False,numChunks=2)
    prompt=f""" 

        “Actúa como un asistente de atención a cliente de startup morgana y experto en hipotecas,
        cuando sea oportuno sugiere que visiten nuestro cotizador en linea https://morgana.mx/mi_espacio/registra_cotizar/,
        Responde preguntas basado en los siguiente contexto:
        --------------------\n
        {context_text}
        ---------------------\n   
       
        Por favor contesta la siguiente pregunta: {query_text}\n
        Sugiere visiten el cotizador https://morgana.mx/mi_espacio/registra_cotizar/ cuando tenga preguntas de tasas, plazos, intereses y costos\n
        Si la respuesta esta en el contexto contesta en 30 palabras o menos. No menciones la palabra contexto\n
        Si la pregunta es en general de (hipotecas o creditos) y la respuesta no esta en el contexto, utiliza otra información. No menciones la palabra contexto  \n
        Si la pregunta no es en general de (hipotecas  o creditos) y la respuesta no esta en el contexto responde 'no tengo la respuesta a esa pregunta. Por favor contacta a uno de nuestros asesores'\n
        Contesta en español
            
            
            """
    response=get_completion(prompt)
    r=response.strip().strip('.').replace("$","\$")
    st.write(f"""{r}""")



if 'index' not in st.session_state:
	st.session_state['index']=createIndex(reIndex=False)


st.title("Pericles")
valor=st.text_input("Pregunta", key="pregunta")
prev_qry = ""
if (prev_qry != valor):
    prev_qry = valor
    tries=0
    done=False
    while (tries<3 and not done):
        try:
            qryGPT(st.session_state.pregunta)
            done=True
        except:
            #st.write("espere un momento porfavor")
            time.sleep(3)
        tries+=1
#st.button("quibo",on_click=pru)





