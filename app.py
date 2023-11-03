from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import streamlit as st

DATA_PATH='data/'
DB_PATH='db/'

custom_prompt_template=""" Use the following pieces of information to answer the user's question in bullet points.
 If you don't know the answer, please just say that you don't know the answer,don't try to make up an answer.
 Context: {context}
 Question: {question}
 
 Only return the helpful answer below and nothing else.
 Helpful answer:
 
 """
 
# setting prompt
def set_prompt():
     prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
     return prompt

# loading llm
def load_llm():
    llm_llama = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML",model_type="llama",
                    config={'max_new_tokens':256,'temperature':0.5})
    return llm_llama

# creating qa 
def qa_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':"cpu"})
    db = FAISS.load_local(DB_PATH, embeddings)
    llm = load_llm()
    prompt=set_prompt()
    qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                               retriever=db.as_retriever(search_kwargs={'k': 4}),
                               return_source_documents=False, 
                               chain_type_kwargs={"prompt":prompt})
    return qa
print("created model")


# stream_lit front-end
st.title("welcome")
with st.sidebar:
    st.title('🤖💬 KMBR law assistant')
  
    st.success('Proceed to entering your prompt message!', icon='👉')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("ask your query"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        chain=qa_model()
        res=chain({'query':prompt})
        answer=res["result"]
        
        message_placeholder.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})





    
    
