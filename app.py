import streamlit as st
import os
import pdfplumber
import pandas as pd
import time
import pytesseract
import tempfile

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"        # API KEY DONT CHANGE

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css
from transformers import MarianMTModel, MarianTokenizer
from pdf2image import convert_from_path
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"   # DOWNLOAD TESSERACT TO C:\Program Files, PASTE THE PATH HERE

# VALIDATE UPLOADED FILE
def validate_pdf_files(pdf_docs, max_files=3, max_size_mb=5):
    if not pdf_docs:
        return "Please upload at least one PDF file."
    elif len(pdf_docs) > max_files:
        return f"Maximum {max_files} PDF files allowed at a time."
    elif pdf_docs:
        for pdf in pdf_docs:
            if pdf.size > max_size_mb * 1024 * 1024:
                return f"File: {pdf.name} exceeds the {max_size_mb}MB size limit. \n\nPlease upload smaller files."
            if not pdf.name.endswith('.pdf'):
                return f"File: {pdf.name} is not a PDF. \n\nPlease upload only PDF files."
    return None


# TO EXTRACT RAW TEXT FROM PDF
def get_pdf_text_and_tables(pdf_docs):
    text = ""
    tables = []

    for pdf in pdf_docs:
        try:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf.read())
                temp_pdf_path = temp_pdf.name  # Temporary file path
            
            with pdfplumber.open(temp_pdf_path) as pdf_reader:
                # Loop through all pages in the PDF
                for page_number, page in enumerate(pdf_reader.pages, start=1):
                    page_text = page.extract_text()  # Extract text from the page
                    
                    if page_text and page_text.strip():  # Only add non-empty text
                        text += page_text
                    else:
                        st.warning(f"No readable text on page {page_number} of {pdf.name}. Attempting OCR...")
                        # Use OCR as fallback only for empty pages
                        poppler_path = r"C:\Users\Nicholas\Downloads\-Interactive-Chat-App-main\poppler-24.08.0\Library\bin"    # DOWNLOAD POPPLER AND MOVE INTO THIS FOLDER, PASTE PATH
                        images = convert_from_path(temp_pdf_path, poppler_path=poppler_path, first_page=page_number, last_page=page_number)
                        
                        for image in images:
                            ocr_text = pytesseract.image_to_string(image, lang='eng')
                            if ocr_text.strip():  # Only add non-empty OCR text
                                text += ocr_text
                    
                    # Extract tables if available
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)

        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
    
    if not text.strip() and not tables:
        st.error("No readable content found in the uploaded PDF(s). Please try a different file.")
        return "", []
    
    return text, tables



# TO CHUNK RAW TEXT
def get_text_chunks(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# TO CREATE VECTOR STORE
@st.cache_data
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")     # DONT CHANGE
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# TO GET CONVERSATION CHAIN
def get_conversation_chain(vectorstore, selected_model):
    llm = HuggingFaceHub(
         repo_id=selected_model,
        model_kwargs={"temperature": 0.5})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory)
    return conversation_chain


# DISPLAY AND VISUALIZE TABLE
def display_tables(tables):
    for i, table in enumerate(tables):
        try:
            # CHECK FOR TABLE STRUCTURE
            if len(table) > 1:
                df = pd.DataFrame(table[1:], columns=table[0])
            else:
                df = pd.DataFrame(table)

            # CHECK IF TABLE EMPTY
            if df.empty:
                st.warning(f"Table {i+1} is empty.")
                continue

            # DISPLAY TABLE
            st.write(f"Table {i+1}:")
            st.dataframe(df)

            # TRY TO CONVERT ALL COLUMN INTO NUMERIC
            df_converted = df.apply(pd.to_numeric, errors='coerce')

            # CHECK COL FOR NUMERIC DATA
            numeric_cols = df_converted.columns[df_converted.notna().any()]
            numeric_cols = numeric_cols[df_converted[numeric_cols].applymap(lambda x: isinstance(x, (int, float))).all()]

            if numeric_cols.empty:
                # IF NO VALID NUMERIC COL   TREAT AS STRING TABLE
                continue

            # VISUALIZE WHEN VALID NUMERIC COL
            st.write("Visualization")
            # BAR CHART
            st.bar_chart(df_converted[numeric_cols])

        except Exception as e:
            st.warning(f"Error processing Table {i+1}: {str(e)}")


# TO HANDLE USER INPUT
def handle_userinput(user_question):
    # CHECK IF A PDF IS PROCESSED
    if not st.session_state.conversation:
        st.error("Please upload and process a PDF first!")
        return
    
    if user_question.lower() == "summary":          # SUMMARY
        if "text_chunks" in st.session_state:
            with st.spinner("Generating summary..."):
                # USE CACHED IF AVAILABLE
                if "summary" in st.session_state and st.session_state.summary:      
                    st.write("### Cached Summary:")
                    st.write(st.session_state.summary)
                else:
                    start_time = time.time()
                    summary = summarize_text(st.session_state.text_chunks)
                    time_taken = time.time() - start_time
                    st.session_state.summary = summary
                    st.write("### Summary:")
                    st.write(summary)
                    st.success(f"Summary generated successfully in {time_taken:.2f} seconds!")
    elif user_question.lower() == "translate":      # TRANSLATION
        with st.spinner("Translating text to Indonesian..."):
            # USE CACHED IF AVAILABLE
            if "indonesian_translation" in st.session_state and st.session_state.indonesian_translation: 
                st.write("### Cached Translated Text (Indonesian):")
                st.write(st.session_state.indonesian_translation)
            elif "text_chunks" in st.session_state:
                start_time = time.time()
                indonesian_translation = translate_large_text_to_indonesian(st.session_state.text_chunks)
                time_taken = time.time() - start_time
                st.session_state.indonesian_translation = indonesian_translation
                st.write("### Translated Text (Indonesian):")
                st.write(indonesian_translation)
                st.success(f"Translation generated successfully in {time_taken:.2f} seconds!")
            else:
                st.warning("Please process the PDF first and enable translation.")
    else:
        # RESPONSE TO USER INPUT
        start_time = time.time()
        response = st.session_state.conversation({'question':user_question})
        st.session_state.chat_history = response['chat_history']
        time_taken = time.time() - start_time
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                with st.chat_message("user"):
                    st.write(message.content)  
            else:
                with st.chat_message("assistant"):
                    st.write(message.content)
        st.success(f"Responses generated successfully in {time_taken:.2f} seconds!")


# RESET CONVERSATION
def reset_conversation():
    if st.sidebar.button("Reset Conversation"):
        st.session_state.conversation = None
        st.session_state.chat_history = None
        st.success("Conversation reset successfully.")


# DOWNLOAD CHAT HISTORY
def save_chat_history(chat_history):
    if chat_history:
        # FORMAT CHAT HISTORY
        history_text = "\n".join(
            f"User: {msg.content}" if i % 2 == 0 else f"Assistant: {msg.content}\n"
            for i, msg in enumerate(chat_history)
        )
        # DOWNLOAD BUTTON
        st.download_button(
            label="Download Chat History",
            data=history_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )


# SUMMARIZE PDF
def summarize_text(text_chunks):
    llm = HuggingFaceHub(
        repo_id="facebook/bart-large-cnn",      # LLM FOR SUMMARY
        model_kwargs={"temperature": 0.5})
    summaries = []
    for i, chunk in enumerate(text_chunks):
        with st.spinner(f"Summarizing chunk {i+1}/{len(text_chunks)}..."):
            summary = llm(chunk)
            summaries.append(summary)
    # Combine summaries
    final_summary = "\n\n".join(summaries)
    return final_summary


# LARGE TEXT TO INDONESIAN
def translate_large_text_to_indonesian(text_chunks):
    translated_chunks = []
    for chunk in text_chunks:
        translated_chunks.append(translate_to_indonesian(chunk))
    return " ".join(translated_chunks)


# TRANSLATE TO INDONESIAN
def translate_to_indonesian(text):
    # Load English to Indonesian model
    model_name = "Helsinki-NLP/opus-mt-en-id"       # SELECTED MODEL FOR TRANSLATION EN-ID
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Perform translation
    translated = model.generate(**inputs)

    # Decode the translated text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text





# MAIN FUNCTION
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF(s)", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # MAIN HEADER & INITIAL CHAT
    st.header("Chat with ChillBot 🤖")
    st.subheader("Upload PDF(s) and Chat to Us ;)")
    with st.chat_message("user"):
        st.write("Hello ChillBot👋")
    with st.chat_message("assistant"):
        st.write("Hello Chill Guy! Feel free to upload your PDF(s) and ask questions.")

    # SIDEBAR
    with st.sidebar:
        with st.expander("🚀 **Model Configuration**", expanded=False):
            llm_model = st.selectbox(
                "Select an LLM Model:",
                ["google/flan-t5-large", "Qwen/QwQ-32B-Preview", "HuggingFaceH4/zephyr-7b-alpha"],
            )
            if llm_model == "google/flan-t5-large":
                st.success("✅ This model is optimized for **Q&A tasks**.")
            else:
                st.warning("✨ This model works best for **text generation**.")

    # USER INPUT    MUST PUT ABOVE OTHER PROCESSING TO ENSURE CHAT HISTORY WORKS
    user_question = st.chat_input("Ask a question about your document(s):")
    if user_question:
        handle_userinput(user_question)

    # SIDEBAR
    with st.sidebar:
        st.divider()
        pdf_docs = st.file_uploader("📄 PDF Upload", accept_multiple_files=True)
        
        # VALIDATION FOR UPLOADED FILE 
        validation_error = validate_pdf_files(pdf_docs)
        if validation_error:
            st.warning(validation_error)
            pdf_docs = None

        # SLIDERS FOR CHUNK SIZE AND OVERLAP
        chunk_size = st.sidebar.slider("Set chunk size", min_value=500, max_value=2000, value=1000, step=100, disabled=not pdf_docs)
        chunk_overlap = st.sidebar.slider("Set chunk overlap", min_value=100, max_value=500, value=200, step=50, disabled=not pdf_docs)
        if chunk_overlap >= chunk_size:
            st.error("Chunk overlap must be smaller than chunk size. Please adjust the slider values.")
            return

        # PROCESS BUTTON
        process_button = st.button("Process", disabled=not pdf_docs)
        
        # RESET CONVERSATION & SAVE CHAT HISTORY    AVAILABLE WHEN CHAT
        if st.session_state.chat_history:
            reset_conversation()
            save_chat_history(st.session_state.chat_history)

    # PROCESS PDF
    if process_button:
        with st.spinner("Processing PDF(s)..."):
            start_time = time.time()

            # CLEAR PREVIOUS STATE
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.session_state.indonesian_translation = None  
            st.session_state.summary = None  
            st.session_state.text_chunks = None  

            # GET PDF TEXT AND TABLES
            raw_text, tables = get_pdf_text_and_tables(pdf_docs)

            if raw_text.strip():
                # GET TEXT CHUNKS
                text_chunks = get_text_chunks(raw_text, chunk_size, chunk_overlap)

                # Store the text chunks in session_state for future use (e.g., summarization or translation)
                st.session_state.text_chunks = text_chunks

                # CREATE VECTOR STORE
                vectorstore = get_vectorstore(text_chunks)

                # CREATE CONVERSATION CHAIN
                st.session_state.conversation = get_conversation_chain(vectorstore, llm_model)

                # DISPLAY TABLES (if extracted)
                display_tables(tables)

                # PROCESSING SUMMARY DATA
                total_text_length = len(raw_text.strip())  # Total extracted text length
                num_pages = sum(len(PdfReader(pdf).pages) for pdf in pdf_docs)  # Total number of pages
                num_tables = len(tables)  # Number of extracted tables

                # DISPLAY PROCESSING SUMMARY
                time_taken = time.time() - start_time
                st.write(f"✅ **Processing Summary**:")
                st.write(f"- Total extracted text length: **{total_text_length} characters**")
                st.write(f"- Number of pages processed: **{num_pages}**")
                st.write(f"- Number of tables extracted: **{num_tables}**")
            else:
                return
            st.success(f"PDF(s) processed successfully in {time_taken:.2f} seconds.")
            st.info("Enter [Summary] to get a summary of your PDF(s).\n\nEnter [Translate] to translate PDF(s) into Indonesian.\n\nEnter [your_question] to ask anything about the PDF(s).")

if __name__ == '__main__':
    main()
