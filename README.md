# Project Title: CSV Data Interaction with RAG Chatbot and REACT Agent

# Description:
This project focuses on enabling efficient interaction with CSV data using advanced language technologies. It includes two versions: a RAG (Retrieval Augmented Generation) chatbot and a REACT (ReAct Prompting) Agent. Both versions offer unique ways to extract insights and perform tasks with CSV data.

## Version 1: RAG Chatbot
![jumpstart-fm-rag](https://github.com/Kirouane-Ayoub/CSV-Data-Interaction/assets/99510125/e52ec4ae-7f8a-4cc4-9638-5f80f45a101a)

The RAG chatbot leverages LangChain, ChromaDB, and Hugging Face's large language models to provide an intuitive interface for users to inquire about and analyze CSV data.
It employs Retrieval Augmented Generation (RAG) to find answers within the CSV dataset and generate concise responses, making data exploration effortless.
### Retrieval Augmented Generation (RAG) : 
+ Check this YouTube Video: **https://www.youtube.com/watch?v=T-D1OfcDW1M**

## Version 2: REACT Agent

The REACT Agent is built using REACT prompting, a few-shot task-solving approach. It utilizes human-written text reasoning traces, actions, and environment observations to interact with CSV data.
This version offers a flexible and powerful way to perform various tasks with CSV data, from question answering to data manipulation, all through natural language prompts.

<img width="1535" alt="diagram" src="https://github.com/Kirouane-Ayoub/CSV-Data-Interaction/assets/99510125/a307839b-29d3-4cc1-984e-2b8340643c1a">

### REACT prompting (Reasoning and Acting) :
+ Check this YouTube Video: **https://www.youtube.com/watch?v=Eug2clsLtFs&t**
## Key Features:

+ User-friendly GUI for both versions, making data interaction accessible to all users.
+ CSV data analysis, querying, and manipulation capabilities.
+ Support for various Hugging Face models for advanced language understanding.
+ Few-shot task-solving capabilities in the REACT Agent for diverse data tasks.
## Technologies Used:

+ **LangChain**: Simplifying the integration of large language models.
+ **ChromaDB**: Open-source vector store for storing and retrieving vector embeddings.
+ **Hugging Face Models**: A wide range of pre-trained language models for natural language processing.
+ **REACT Prompting**: An intuitive and flexible approach for few-shot task-solving with natural language.

## Usage : 

```
pip install -r requirements.txt
# REACT Agent
cd Agent
streamlit run CSV_Agent.py
# RAG Chatbot
streamlit run CSV_ChatBot.py
```
