import streamlit as st
import requests
import os
import tempfile
import PyPDF2
import io
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OCR_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    pass

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file without OCR."""
    text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_file_path = temp_file.name
        
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        os.unlink(temp_file_path)  # Delete the temporary file
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_with_ocr_if_available(pdf_file):
    """Extract text from PDF, using OCR if available, otherwise falling back to standard extraction."""
    if not OCR_AVAILABLE:
        st.warning("OCR functionality is not available. Installing dependencies would enable image text extraction.")
        st.info("To enable OCR, install: pip install pytesseract pdf2image pillow")
        st.info("You'll also need to install Poppler on your system.")
        return extract_text_from_pdf(pdf_file)
    
    full_text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_file_path = temp_file.name
        
        # Standard text extraction
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                full_text += text + "\n\n"
        
        # OCR for images in the PDF
        try:
            with st.spinner("Performing OCR on images in PDF..."):
                images = convert_from_path(temp_file_path)
                for i, image in enumerate(images):
                    # Save image temporarily
                    img_path = f"temp_img_{i}.png"
                    image.save(img_path, "PNG")
                    
                    # Perform OCR
                    img_text = pytesseract.image_to_string(Image.open(img_path))
                    if img_text.strip():  # Only add if we got meaningful text
                        full_text += f"\n[Image Content Page {i+1}]: {img_text}\n"
                    
                    # Clean up
                    if os.path.exists(img_path):
                        os.remove(img_path)
        except Exception as ocr_e:
            st.warning(f"OCR processing failed, but basic text extraction succeeded: {ocr_e}")
            st.info("Proceeding with text-only extraction. To enable OCR, ensure Poppler is properly installed.")
        
        os.unlink(temp_file_path)  # Delete the temporary file
        return full_text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""

def create_vectorstore(text):
    """Create a vector store from the paper text."""
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings using SciBERT
    embeddings = HuggingFaceEmbeddings(
        model_name="allenai/scibert_scivocab_uncased",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def evaluate_response(reference, candidate):
    """
    Evaluate the quality of the AI response using BLEU score
    Args:
        reference: The reference text (ground truth)
        candidate: The AI-generated response
    Returns:
        BLEU score
    """
    try:
        # Tokenize the texts (simple word tokenization)
        reference_tokens = [reference.lower().split()]
        candidate_tokens = candidate.lower().split()
        
        # Calculate BLEU score with smoothing
        smooth = SmoothingFunction().method1
        bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smooth)
        
        return bleu_score
    except Exception as e:
        st.warning(f"Error calculating BLEU score: {e}")
        return 0.0

def run_research_assistant():
    st.subheader("📚 Research Paper Assistant")
    st.write("Upload your research paper and ask questions about it.")

    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
    
    if uploaded_file:
        if st.button("Process Paper"):
            with st.spinner("Processing PDF..."):
                paper_text = extract_text_with_ocr_if_available(uploaded_file)
                if not paper_text:
                    st.error("Failed to extract text from the PDF. Please try another file.")
                    return
                
                st.session_state.paper_text = paper_text
                st.success(f"Successfully processed paper ({len(paper_text)} characters)")
                
                with st.spinner("Creating knowledge base with SciBERT embeddings..."):
                    vectorstore = create_vectorstore(paper_text)
                    st.session_state.vectorstore = vectorstore
                    st.success("Paper knowledge base created! You can now ask questions.")
    
    if 'vectorstore' in st.session_state:
        st.write("### Ask Questions About Your Paper")
        question = st.text_input("What would you like to know about this paper?", 
                                placeholder="e.g., What is the main conclusion of this study?")
        
        if question and st.button("Get Answer"):
            with st.spinner("Thinking..."):
                try:
                    # Use Mistral from Ollama
                    llm = Ollama(model="mistral", temperature=0.7)
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 3}
                        )
                    )
                    
                    response = qa_chain.run(question)
                    
                    st.write("### Answer")
                    st.write(response)
                    
                    # Get relevant sections for evaluation
                    docs = st.session_state.vectorstore.similarity_search(question, k=3)
                    contexts = [doc.page_content for doc in docs]
                    combined_context = " ".join(contexts)
                    
                    # Calculate and display BLEU score
                    bleu = evaluate_response(combined_context, response)
                    
                    # Display evaluation metrics
                    st.write("### Response Evaluation")
                    st.write(f"BLEU Score: {bleu:.4f}")
                    
                    # Create color coding based on BLEU score
                    if bleu > 0.5:
                        score_color = "green"
                        quality = "Excellent"
                    elif bleu > 0.3:
                        score_color = "orange"
                        quality = "Good"
                    else:
                        score_color = "red"
                        quality = "Fair"
                    
                    st.markdown(f"<div style='background-color:{score_color}; padding:10px; border-radius:5px;'>"
                               f"<p style='color:white; margin:0;'>Response Quality: {bleu:.4f} ({quality})</p></div>", 
                               unsafe_allow_html=True)
                    
                    with st.expander("See relevant sections from the paper"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Relevant Section {i+1}:**")
                            st.write(doc.page_content)
                            st.write("---")
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    st.info("Make sure Ollama is running with the Mistral model. You can install it with: `ollama pull mistral`")

    if 'vectorstore' not in st.session_state:
        st.info("👆 Upload a research paper (PDF) to get started.")
        
        feature_set = "text extraction"
        if OCR_AVAILABLE:
            feature_set = "text extraction and OCR for images"
            
        st.write(f"""
        **How it works:**
        1. Upload your research paper in PDF format
        2. The system performs {feature_set}
        3. Creates a searchable knowledge base using SciBERT embeddings
        4. Ask specific questions about the paper's content
        5. Get contextual answers using the Mistral language model via Ollama
        6. View quality metrics for the AI's response
        """)
        
        # Display dependency information
        with st.expander("System Requirements"):
            st.markdown("""
            **Core Dependencies:**
            - Python 3.7+
            - Streamlit
            - PyPDF2
            - LangChain
            - FAISS
            - SciBERT embeddings (HuggingFace)
            - Ollama with Mistral model
            
            **For OCR Features (Optional):**
            - pytesseract
            - pdf2image
            - Poppler (system installation required)
            - Tesseract OCR (system installation required)
            
            **For BLEU Score Evaluation:**
            - NLTK
            """)

if __name__ == "__main__":
    # Set up page configuration
    st.set_page_config(page_title="Research Paper Assistant", page_icon="📚", layout="wide")
    
    # Add dependencies check
    try:
        import nltk
        nltk.download('punkt', quiet=True)
    except ImportError:
        st.warning("NLTK not installed. Evaluation metrics might not work properly.")
    
    run_research_assistant()