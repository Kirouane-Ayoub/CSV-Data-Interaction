from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent
import streamlit as st

st.set_page_config(page_title='CSV Chatbot')

def model(temperature ,top_p , model_id_name) :
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
    return llm 
model_id_name = st.sidebar.text_input("Input your  LLM HF model ID")

model_id_name = st.sidebar.text_input("Input your  LLM HF model ID")

temperature = st.sidebar.slider("Select your temperature value : " ,min_value=0.1 ,
                                 max_value=1.0 ,
                                   value=0.5)
top_p = st.sidebar.slider("Select your top_p value : " ,min_value=0.1 ,
                           max_value=1.0 , 
                           value=0.5)

if st.sidebar.button("START") : 
    if model_id_name  : 
        with st.spinner(f"downloading {model_id_name} model ..."):
            llm = model(temperature ,top_p , model_id_name) 

csv_file = st.file_uploader("Please upload your CSV file to start The conversation: " , type=['csv'])
if csv_file : 
    csv_path = csv_file.name
    agent = create_csv_agent(llm, 
                            csv_path, 
                            verbose=True , 
                            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
                            )
    def chatbot_response(input_text):
        return agent.run(input_text) 
    
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hi :hand: Im Your open-source CSV Agent, You can ask any thing about you CSV file content."}]

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