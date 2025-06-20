import streamlit as st
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

groq_api = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("conversation RAG with PDF uploads and chat history")
st.write("upload pdf and chat with its content and much more")

llm = ChatGroq(model="Gemma2-9b-It",api_key=groq_api)
session_id = st.text_input("Session Id",value="default_session")

if 'store' not in st.session_state:
    st.session_state.store={}

uploaded_files=st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_q_system = (
        """
        Given a chat history and the latest user question
        which might reference context in the chat history
        formulate a standalone question which can be understood
        without the chat history. Do not answer the question,
        just reformulate it if needed and otherwise return it as is.
        """
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system),
            MessagesPlaceholder("chat_history"),
            ("user","{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    system_prompt=(
        """
        your are an assistant for question-answering tasks,
        use the following peices of retrieved context to answer
        the question.
        user wants to know about something which is not in the context just write that you dont know that from the context then give answer to that question.
        {context}
        """
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session_id:str)-> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]

    conversation_rag_cahin = RunnableWithMessageHistory(
        rag_chain,get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_area("Write the questions: ")

    if user_input:
        session_history = get_session_history(session_id)
        response = conversation_rag_cahin.invoke(
            {"input":user_input},
            config={
                "configurable":{"session_id":session_id}
            }
        )
        st.success(f"Assistant: {response['answer']}")


