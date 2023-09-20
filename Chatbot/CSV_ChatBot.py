from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title='CSV Chatbot')

def model(temperature ,top_p , model_id_name , embed_model_id ) :
    tokenizer = LlamaTokenizer.from_pretrained(model_id_name)

    base_model = LlamaForCausalLM.from_pretrained(
        model_id_name,
        load_in_4bit=True,
        device_map='auto',
    )
    pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.15
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    #embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    embed_model = HuggingFaceEmbeddings(model_name=embed_model_id)
    return llm , embed_model
model_id_name = st.sidebar.text_input("Input your  LLM HF model ID")
embed_model_id = st.sidebar.text_input("Input your Embeddings HF model ID")

temperature = st.sidebar.slider("Select your temperature value : " ,min_value=0.1 ,
                                 max_value=1.0 ,
                                   value=0.5)
top_p = st.sidebar.slider("Select your top_p value : " ,min_value=0.1 ,
                           max_value=1.0 , 
                           value=0.5)

k_n = st.sidebar.number_input("Enter the number of top-ranked retriever Results:" ,
                             min_value=1 , max_value=5 , value=4)

if st.sidebar.button("START") : 
    if model_id_name  : 
        with st.spinner("downloading (model + tokenizer)..."):
            llm , embed_model = model(temperature ,top_p , model_id_name)


def loadCSV(file_path) : 
  loader = CSVLoader(file_path=file_path)
  data = loader.load()
  return data

csv_ffile = st.file_uploader("Please upload your CSV file to start The conversation: " , type=['csv'])
if csv_ffile and llm and embed_model : 
    with st.spinner("In progress...") :
        data=loadCSV(csv_ffile.name)
        vectordb = Chroma.from_documents(data, embedding=embed_model , persist_directory="DB")

    chain = ConversationalRetrievalChain.from_llm(llm =llm , retriever=vectordb.as_retriever())

    def chatbot_response(input_text):
        return chain(input_text)
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hi :hand: Im Your open-source CSV chatbot, You can Chat with your Data."}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)