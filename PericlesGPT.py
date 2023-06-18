import openai
import os
import pandas as pd
from scipy.spatial.distance import cosine
import re
import json
import time
import streamlit as st
from tenacity import retry, wait_random_exponential, stop_after_attempt
LLAVE="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"

appOn=False

if appOn:
	INDEXGPTDIR="./"
	DATOSGPTDIR="/Users/fesponda/Morgana/gpt/datos/"
	os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
	os.environ["OPENAI_API_KEY"] = LLAVE

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



def pru():
    #st.write(st.session_state['context_text'])
    st.session_state['context_text']=''
    st.session_state.pregunta=''

st.title("Pericles")
if 'context_text' not in st.session_state:
    st.session_state['context_text']=''



valor=st.text_input("Pregunta", key="pregunta")

prev_qry = ""
if (prev_qry != valor):
    prev_qry = valor
    cont=0
    fin=False
    st.session_state['context_text']=st.session_state['context_text']+"\n"+st.session_state.pregunta
    #print(st.session_state['context_text'])
    #context_text=st.session_state.pregunta'
    campos="nombre ,valor del inmueble, enganche, ingresos, email"
    prompt=f""" 

        Actúa como un asistente \
        llena el siguiente Json

        Responde preguntas basado en los siguiente contexto: 
        --------------------\
         {st.session_state['context_text']}
        ---------------------\

        Por favor contesta la siguiente pregunta usando la informacion anterior: "llena el siguiente json con los siguientes campos 
        ---{campos}---  "
        " \
        Para los campos de credito e ingresos usa numeros y no palabras\
        Si algun campo no puedes llenar inserta un -1 \
        Si no encuentas nombre regresa ""\
        No uses informacion fuera del contexto\

            """
    #        Solo pregunta por los campos de la pregunta \
    #        Si algun campo de la pregunta no esta en el contexto, responde con los nombre de los campos que faltan \


    res=get_completion(prompt,model="gpt-3.5-turbo-0613")
    data=re.findall(r'{.*}', str(res).replace("\n",""))[0]
    #print(data)
    try:

        data=dict(eval(str(data)))
        s='Por favor proporcione lo siguientes valores: '
        l=len(s)
        #st.write('uno')
        for k in data.keys():
            #st.write(k)

            if data[k]=="-1" or data[k]==-1 or len(str(data[k]))==0 : #len(str(data[k]))<3 or 'contexto' in str(data[k]) or 'info' in str(data[k]) or 'dato' in str(data[k]):
                s+=k+", "
        if len(s)>l:
            st.write(s[0:-2])

        else:
            st.write(data)

        st.session_state['context_text']=str(data)
        fin =True
    except:

        st.write('no',res)
        pass


    st.button("reset",on_click=pru)





