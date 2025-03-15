#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
"""
Retrieval-Augmented Generation (RAG) for Financial Questions
"""
 
import os
import sys
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import time
from typing import List, Dict, Any, Tuple
 
# Completely disable Streamlit's file watcher to prevent issues with torch.classes
# This needs to be done before importing streamlit
os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"
 
# Now it's safe to import torch and streamlit
import torch
from tqdm.auto import tqdm
import streamlit as st
 
# Additional protection: monkey patch torch._classes.__getattr__ to prevent errors
try:
    import torch._classes
    original_getattr = torch._classes.__getattr__
    
    def safe_getattr(self, attr):
        if attr == "__path__":
            return None
        return original_getattr(self, attr)
    
    torch._classes.__getattr__ = safe_getattr
except Exception:
    pass  # If we can't patch it, just continue
 
# For document loading and processing
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
 
# For embeddings and vector stores
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
 
# For LLM
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
 
# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
 
# Check for GPU/Metal availability
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
 
# Create directories for data and models
data_dir = Path("financial_data")
data_dir.mkdir(exist_ok=True)
 
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
 
vector_store_path = "financial_vector_store"
 
# Function to download financial statements
def download_financial_statement(url: str, filename: str) -> None:
    """Download a financial statement PDF from a URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(data_dir / filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}. Error: {str(e)}")
 
# Function to download a model if it doesn't exist
def download_model(url: str, filename: str) -> Path:
    """Download a model file from a URL"""
    model_path = models_dir / filename
    if not model_path.exists():
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(model_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size//8192):
                    f.write(chunk)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}. Error: {str(e)}")
    else:
        print(f"{filename} already exists")
    return model_path
 
# Load PDF documents
def load_documents() -> List[Any]:
    """Load all PDF documents from the data directory"""
    documents = []
    for pdf_file in data_dir.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
            print(f"Loaded {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {str(e)}")
    return documents
 
# Split documents into chunks
def split_documents(documents: List[Any]) -> List[Any]:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", ",", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
 
# Financial RAG System
class FinancialRAG:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the RAG system"""
        if self.is_initialized:
            return True
            
        print("Initializing Financial RAG system...")
        
        # Download financial statements
        financial_statements = {
            "apple_2022.pdf": "https://s2.q4cdn.com/470004039/files/doc_financials/2022/q4/_10-K-2022-(As-Filed).pdf",
            "apple_2023.pdf": "https://s2.q4cdn.com/470004039/files/doc_financials/2023/q4/FY23_10K_As-Filed.pdf"
        }
        
        for filename, url in financial_statements.items():
            if not (data_dir / filename).exists():
                download_financial_statement(url, filename)
        
        # Load documents
        documents = load_documents()
        if not documents:
            print("No documents loaded.")
            return False
        
        print(f"Loaded {len(documents)} document pages")
        
        # Split documents into chunks
        chunks = split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )
        
        # Create vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print("Created vector store")
        
        # Save vector store for future use
        self.vector_store.save_local(vector_store_path)
        print(f"Saved vector store to {vector_store_path}")
        
        # Download model
        model_url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        model_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        model_path = download_model(model_url, model_filename)
        
        # Initialize LLM with Metal support if available
        self.llm = LlamaCpp(
            model_path=str(model_path),
            temperature=0.1,
            max_tokens=1024,
            top_p=0.95,
            n_ctx=2048,
            n_gpu_layers=-1,  # Use Metal acceleration if available
            verbose=False
        )
        print("Initialized language model")
        
        # Create QA chain
        template = """
        You are a financial analyst assistant that helps answer questions about Apple's financial statements.
        
        Instructions:
        1. Use ONLY the following context to answer the question.
        2. Provide a direct, concise answer without repeating the question.
        3. If the context doesn't contain the information needed, say "I don't have enough information to answer this question."
        4. Do NOT ask new questions in your response.
        5. Include specific numbers and financial data when available.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        print("Created QA chain")
        
        self.is_initialized = True
        return True
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a simple confidence score calculation"""
        if not self.is_initialized:
            raise ValueError("RAG system not initialized")
        
        # Query the RAG system
        result = self.qa_chain({"query": question})
        
        # Calculate a simple confidence score based on the retrieved documents
        source_docs = result.get("source_documents", [])
        
        # Simple confidence calculation based on number of documents and content relevance
        if len(source_docs) == 0:
            confidence_score = 0.0
        else:
            # Check if key financial terms are in the retrieved documents
            financial_terms = [
                "revenue", "profit", "margin", "earnings", "dividend",
                "balance sheet", "income statement", "cash flow",
                "fiscal", "quarter", "annual report", "10-K", "10-Q"
            ]
            
            # Count how many financial terms appear in the retrieved documents
            term_matches = 0
            for doc in source_docs:
                content = doc.page_content.lower()
                for term in financial_terms:
                    if term in content:
                        term_matches += 1
            
            # Calculate confidence based on term matches and number of documents
            doc_score = min(1.0, len(source_docs) / 3.0)  # Normalize by expected number of docs
            term_score = min(1.0, term_matches / (len(financial_terms) * 0.5))  # Only need half the terms
            
            # Combine scores (weighted average)
            confidence_score = (doc_score * 0.4) + (term_score * 0.6)
            
            # Adjust based on question type and content
            question_lower = question.lower()
            
            # Check for direct financial metrics questions
            is_direct_metric_question = (
                ("what" in question_lower or "how much" in question_lower) and
                any(term in question_lower for term in ["revenue", "profit", "earnings", "income", "sales"])
            )
            
            # Check if the answer contains specific financial figures (numbers with $ or million/billion)
            answer = result.get("result", "").lower()
            has_financial_figures = (
                any(char.isdigit() for char in answer) and
                ("$" in answer or "million" in answer or "billion" in answer)
            )
            
            if is_direct_metric_question:
                # High confidence boost for direct financial metrics questions
                confidence_score = min(1.0, confidence_score * 2.0)
                
                # Additional boost if the answer contains specific figures
                if has_financial_figures:
                    confidence_score = min(1.0, confidence_score * 1.25)
            
            elif "compare" in question_lower or "trend" in question_lower:
                # Lower confidence for comparison/trend questions
                confidence_score = confidence_score * 0.8
        
        # Add confidence score to result
        result["confidence_score"] = confidence_score
        result["confidence_level"] = self._get_confidence_level(confidence_score)
        
        return result
    
    def _get_confidence_level(self, score: float) -> str:
        """Convert numerical confidence score to descriptive level"""
        if score >= 0.8:
            return "High"
        elif score >= 0.5:
            return "Medium"
        elif score >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def validate_query(self, query: str) -> tuple:
        """Validate the query with improved guardrails"""
        # Check if query is empty
        if not query.strip():
            return False, "Query cannot be empty."
        
        # Check if query is too short
        if len(query.split()) < 3:
            return False, "Query is too short. Please provide a more detailed question."
        
        query_lower = query.lower()
        
        # List of common non-financial topics to explicitly reject
        non_financial_topics = [
            "capital of", "president of", "population of", "weather in", "recipe for",
            "how to cook", "who won", "when was", "where is", "tallest mountain",
            "deepest ocean", "largest country", "smallest country", "prime minister",
            "king of", "queen of", "movie", "actor", "actress", "song", "music",
            "book", "novel", "author", "painter", "artist"
        ]
        
        # Check for explicit non-financial topics
        for topic in non_financial_topics:
            if topic in query_lower:
                return False, f"This question about '{topic}' is not related to financial statements. Please ask a finance-related question."
        
        # Check if query is related to finance
        financial_terms = [
            "revenue", "profit", "margin", "eps", "earnings", "dividend", "balance", "sheet",
            "income", "statement", "cash", "flow", "financial", "fiscal", "quarter", "annual",
            "report", "stock", "share", "market", "capital expenditure", "asset", "liability", "equity",
            "expense", "cost", "tax", "investment", "debt", "credit", "loan", "interest", "rate",
            "apple", "company", "business", "sales", "growth", "decline", "increase", "decrease"
        ]
        
        # Check for company-specific terms
        company_terms = ["apple", "company", "business", "corporation", "firm", "enterprise"]
        has_company_term = any(term in query_lower for term in company_terms)
        
        # Check for financial terms
        has_financial_term = any(term in query_lower for term in financial_terms)
        
        # Require either a company term AND a financial term, or a very specific financial term
        specific_financial_terms = ["revenue", "profit", "earnings", "balance sheet", "income statement", "cash flow"]
        has_specific_financial_term = any(term in query_lower for term in specific_financial_terms)
        
        if not (has_specific_financial_term or (has_company_term and has_financial_term)):
            return False, "Query does not appear to be related to financial statements. Please ask a finance-related question about Apple or include specific financial terms."
        
        return True, "Valid query"
 
# Streamlit UI
def main():
    st.title("Financial RAG System")
    st.write("Ask questions about Apple's financial statements from 2022-2023")
    
    # Initialize the RAG system
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = FinancialRAG()
    
    rag_system = st.session_state.rag_system
    
    # Initialize if not already done
    if not rag_system.is_initialized:
        with st.spinner("Initializing the system... This may take a few minutes."):
            success = rag_system.initialize()
            if not success:
                st.error("Failed to initialize the RAG system.")
                return
    
    # Initialize session state for selected question if it doesn't exist
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = ""
    
    # Example test questions
    st.subheader("Example Test Questions:")
    test_questions = {
        "High-confidence financial question": "What was Apple's total revenue in 2022?",
        "Low-confidence financial question": "How did Apple's R&D spending in 2022 compare to previous years?",
        "Irrelevant question (guardrail test)": "What is the capital of France?"
    }
    
    # Function to set the selected question
    def set_question(question):
        st.session_state.selected_question = question
    
    # Create columns for the test question buttons
    cols = st.columns(3)
    
    # Add buttons for each test question
    for i, (question_type, question) in enumerate(test_questions.items()):
        cols[i].button(
            f"Test: {question_type}",
            key=f"btn_{i}",
            on_click=set_question,
            args=(question,)
        )
    
    # Form for user query
    with st.form(key="query_form"):
        # Input for user query
        query = st.text_input(
            "Enter your question:",
            value=st.session_state.selected_question
        )
        
        # Submit button
        submit_button = st.form_submit_button(label="Submit")
        
        if submit_button and query:
            # Validate query
            is_valid, message = rag_system.validate_query(query)
            
            if not is_valid:
                st.error(message)
                st.info("Guardrail activated: This question was rejected because it doesn't appear to be related to financial statements.")
            else:
                with st.spinner("Processing your question..."):
                    try:
                        # Query the RAG system
                        start_time = time.time()
                        result = rag_system.query(query)
                        end_time = time.time()
                        
                        # Display the answer
                        st.subheader("Answer:")
                        st.write(result["result"])
                        
                        # Display confidence information
                        confidence_col1, confidence_col2 = st.columns(2)
                        
                        # Display confidence score with color coding
                        confidence_score = result["confidence_score"]
                        confidence_level = result["confidence_level"]
                        
                        # Color coding based on confidence level
                        if confidence_level == "High":
                            confidence_color = "green"
                        elif confidence_level == "Medium":
                            confidence_color = "orange"
                        elif confidence_level == "Low":
                            confidence_color = "red"
                        else:
                            confidence_color = "red"
                        
                        confidence_col1.metric(
                            "Confidence Score",
                            f"{confidence_score:.2f}",
                            delta=None
                        )
                        
                        confidence_col2.markdown(
                            f"<h3 style='color: {confidence_color};'>Confidence Level: {confidence_level}</h3>",
                            unsafe_allow_html=True
                        )
                        
                        # Display processing time
                        st.write(f"Processing time: {end_time - start_time:.2f} seconds")
                        
                        # Display source documents
                        st.subheader("Source Documents:")
                        for i, doc in enumerate(result["source_documents"][:2]):
                            with st.expander(f"Document {i+1}"):
                                st.write(doc.page_content[:300] + "...")
                                st.write(f"Source: {doc.metadata}")
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
 
if __name__ == "__main__":
    main()
 