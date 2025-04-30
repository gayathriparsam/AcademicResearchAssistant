 Academic Research Assistant

Academic Research Assistant is an AI-powered web application that streamlines the academic research process. 
Built with a user-friendly Streamlit interface and integrated with powerful NLP models (like SciBERT, BART, and Mistral), this tool simplifies tasks such as finding references, summarizing papers, identifying research gaps, and providing structured writing guidance.

---

 Features

- Research Gap Finder  
  Analyze recent papers to identify underexplored areas and generate potential research ideas.

- Finding References  
  Search academic databases (arXiv, Semantic Scholar, Crossref) and rank papers by relevance, citations, and recency.

- Paper Summarizer  
  Summarize PDFs or paper URLs section-by-section for quick insights using BART-large-CNN.

- Clear Doubts  
  Upload a paper and ask questions to get context-aware answers powered by SciBERT and Mistral.

- Writing Guidance  
  Receive customized suggestions for refining research topics, building strong paper structures, and improving writing quality.

---

 Technology Stack

- Frontend: Streamlit
- Backend: Python
- Machine Learning Models: SciBERT, BART-large-CNN, Mistral
- APIs: arXiv, Semantic Scholar, Crossref
- Libraries: Hugging Face Transformers, KeyBERT, PCA, BLEU Score Evaluation

---

 Installation Instructions

1. Clone the repository:
   git clone https://github.com/yourusername/AcademicResearchAssistant.git
2. Navigate to the project folder:
   cd AcademicResearchAssistant
3. Install the required packages:
   pip install -r requirements.txt
4. Run the Streamlit app:
   streamlit run app.py
