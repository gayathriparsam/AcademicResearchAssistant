import streamlit as st
import requests
import json
import re
from transformers import pipeline
from bs4 import BeautifulSoup
import PyPDF2
import io
from datetime import datetime
import numpy as np
from rouge_score import rouge_scorer

class PaperSource:
    def search(self, query, limit=5):
        pass
    
    def get_paper(self, paper_id):
        pass

class ArxivSource(PaperSource):
    def search(self, query, limit=5):
        base_url = "http://export.arxiv.org/api/query?"
        search_query = f"search_query=all:{query}&start=0&max_results={limit}"
        response = requests.get(base_url + search_query)
        
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.content, 'xml')
        entries = soup.find_all('entry')
        
        results = []
        for entry in entries:
            paper_id = entry.id.text.split('/')[-1]
            title = entry.title.text.replace('\n', ' ').strip()
            abstract = entry.summary.text.replace('\n', ' ').strip()
            authors = [author.name.text for author in entry.find_all('author')]
            published = entry.published.text
            
            results.append({
                'id': paper_id,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'published': published,
                'source': 'arxiv',
                'url': entry.id.text
            })
        
        return results
    
    def get_paper(self, paper_id):
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        response = requests.get(pdf_url)
        
        if response.status_code != 200:
            return None
        
        return response.content

class SemanticScholarSource(PaperSource):
    def search(self, query, limit=5):
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,authors,year,url,externalIds"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        results = []
        
        for paper in data.get('data', []):
            paper_id = paper.get('paperId')
            arxiv_id = paper.get('externalIds', {}).get('arxiv')
            
            results.append({
                'id': paper_id,
                'arxiv_id': arxiv_id,
                'title': paper.get('title'),
                'abstract': paper.get('abstract', ''),
                'authors': [author.get('name') for author in paper.get('authors', [])],
                'published': paper.get('year'),
                'source': 'semantic_scholar',
                'url': paper.get('url')
            })
        
        return results
    
    def get_paper(self, paper_id):
        # For Semantic Scholar, we'll try to get the PDF if available
        # Otherwise, we'll return None and let the app handle it
        paper_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=openAccessPdf"
        response = requests.get(paper_url)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        pdf_url = data.get('openAccessPdf', {}).get('url')
        
        if pdf_url:
            pdf_response = requests.get(pdf_url)
            if pdf_response.status_code == 200:
                return pdf_response.content
        
        return None

class CrossrefSource(PaperSource):
    def search(self, query, limit=5):
        url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": limit,
            "sort": "relevance"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        results = []
        
        for item in data.get('message', {}).get('items', []):
            if item.get('type') not in ['journal-article', 'proceedings-article']:
                continue
                
            doi = item.get('DOI')
            title = item.get('title', [''])[0] if item.get('title') else ''
            
            authors = []
            for author in item.get('author', []):
                name_parts = []
                if author.get('given'):
                    name_parts.append(author.get('given'))
                if author.get('family'):
                    name_parts.append(author.get('family'))
                authors.append(' '.join(name_parts))
            
            published = item.get('created', {}).get('date-parts', [['']])[0][0]
            
            results.append({
                'id': doi,
                'title': title,
                'abstract': '',  # Crossref doesn't typically provide abstracts
                'authors': authors,
                'published': published,
                'source': 'crossref',
                'url': f"https://doi.org/{doi}" if doi else ''
            })
        
        return results
    
    def get_paper(self, paper_id):
        # For Crossref, we'll try to resolve the DOI and get the PDF
        # This is a simplified approach and may not work for all publishers
        doi_url = f"https://doi.org/{paper_id}"
        headers = {
            'Accept': 'application/pdf'
        }
        
        try:
            response = requests.get(doi_url, headers=headers, allow_redirects=True)
            if response.status_code == 200 and response.headers.get('Content-Type') == 'application/pdf':
                return response.content
        except:
            pass
        
        return None

class PaperSummarizer:
    def __init__(self):
        # Initialize with a smaller model that's more stable for section summarization
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def extract_sections(self, text):
        # Improved section extraction with better pattern matching
        section_patterns = [
            r'(?:\n|\r\n|^)(\d+\.\s*\w[\w\s]*?)(?=\n|\r\n)',  # Numbered sections like "1. Introduction"
            r'(?:\n|\r\n|^)([A-Z][A-Z\s]+)(?=\n|\r\n)',       # ALL CAPS sections
            r'(?:\n|\r\n|^)(Abstract|Introduction|Related Work|Background|Methodology|Experiments|Results|Discussion|Conclusion|References)(?=\n|\r\n)',  # Common section names
        ]
        
        # First, try to find section headers
        section_headers = []
        section_positions = []
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text):
                header = match.group(1).strip()
                if header and len(header) < 100:  # Avoid matching entire paragraphs
                    section_headers.append(header)
                    section_positions.append(match.start())
        
        # Sort sections by their position in the text
        if section_headers and section_positions:
            sorted_sections = sorted(zip(section_headers, section_positions), key=lambda x: x[1])
            section_headers = [s[0] for s in sorted_sections]
            section_positions = [s[1] for s in sorted_sections]
        
        # If no headers found, chunk the text by paragraphs
        if not section_headers:
            paragraphs = re.split(r'\n\s*\n', text)
            sections = []
            
            current_chunk = ""
            chunk_count = 0
            for i, para in enumerate(paragraphs):
                current_chunk += para + "\n\n"
                
                # Create manageable chunks of text
                if len(current_chunk) > 1000 or i == len(paragraphs) - 1:
                    if chunk_count == 0:
                        sections.append(("Introduction", current_chunk))
                    elif i == len(paragraphs) - 1:
                        sections.append(("Conclusion", current_chunk))
                    else:
                        sections.append((f"Section {chunk_count+1}", current_chunk))
                    current_chunk = ""
                    chunk_count += 1
            
            return sections
        
        # If headers found, extract text between headers
        sections = []
        for i, header in enumerate(section_headers):
            if i < len(section_headers) - 1:
                next_pos = section_positions[i + 1]
                section_text = text[section_positions[i]:next_pos]
            else:
                section_text = text[section_positions[i]:]
            
            # Extract actual content (remove the header from the content)
            header_end = section_text.find('\n')
            if header_end != -1:
                content = section_text[header_end:].strip()
            else:
                content = section_text.strip()
            
            sections.append((header, content))
        
        return sections
    
    def summarize_section(self, section_text, max_length=150):
        # Clean up the text
        text = section_text.strip()
        
        # Skip empty sections
        if not text or len(text.split()) < 20:
            return "Section is too short to summarize."
        
        # Limit input length to avoid model errors
        max_input_length = 1024
        if len(text.split()) > max_input_length:
            text = ' '.join(text.split()[:max_input_length])
        
        try:
            # Handle the summarization more safely with better error handling
            chunks = [text[i:i+512] for i in range(0, len(text), 512)]
            summaries = []
            
            for chunk in chunks[:3]:  # Limit to first 3 chunks to avoid excessive processing
                if len(chunk.split()) < 20:
                    continue
                    
                try:
                    summary = self.summarizer(chunk, max_length=max_length//len(chunks[:3]), 
                                            min_length=30, do_sample=False)[0]['summary_text']
                    summaries.append(summary)
                except Exception as e:
                    st.warning(f"Error summarizing chunk: {str(e)}")
                    # Fall back to extractive summarization when generative fails
                    sentences = chunk.split('. ')
                    if len(sentences) > 3:
                        summaries.append('. '.join(sentences[:3]) + '.')
            
            if summaries:
                return ' '.join(summaries)
            else:
                # Fallback to simple extractive summarization
                sentences = text.split('. ')
                return '. '.join(sentences[:3]) + '.'
                
        except Exception as e:
            # Emergency fallback - just return the first few sentences
            sentences = text.split('. ')
            if sentences:
                return '. '.join(sentences[:3]) + '.'
            return f"Could not summarize section: {str(e)}"
    
    def calculate_rouge_scores(self, summary, reference):
        """Calculate ROUGE scores between summary and reference text"""
        scores = self.rouge_scorer.score(reference, summary)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def summarize_paper(self, pdf_content):
        # Extract text from PDF with better error handling
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                try:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n\n"
                except Exception as e:
                    st.warning(f"Error extracting text from page: {str(e)}")
            
            if not text:
                st.error("Could not extract text from the PDF")
                return [("Error", "Failed to extract text from the PDF.")]
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return [("Error", f"Failed to process the PDF: {str(e)}")]
        
        # Extract sections with error handling
        try:
            sections = self.extract_sections(text)
        except Exception as e:
            st.error(f"Error extracting sections: {str(e)}")
            # Fallback to simple paragraphs
            paragraphs = text.split('\n\n')
            sections = [(f"Section {i+1}", p) for i, p in enumerate(paragraphs) if len(p.strip()) > 100]
            if not sections:
                return [("Error", "Failed to identify sections in the paper.")]
        
        # Summarize each section with proper error handling
        summaries = []
        for section_title, section_text in sections:
            try:
                summary = self.summarize_section(section_text)
                
                # Calculate ROUGE scores comparing summary to original text
                rouge_scores = self.calculate_rouge_scores(summary, section_text)
                
                summaries.append((section_title, summary, rouge_scores, section_text))
            except Exception as e:
                # Provide a graceful fallback for failed summaries
                st.warning(f"Error summarizing section '{section_title}': {str(e)}")
                first_sentences = '. '.join(section_text.split('. ')[:3])
                if first_sentences:
                    summaries.append((section_title, first_sentences + '...', None, section_text))
                else:
                    summaries.append((section_title, "Summary unavailable.", None, section_text))
        
        return summaries

def run_summarization_tool():
    st.title("📚 Research Paper Summarizer")
    
    # Initialize session state
    if "paper_summaries" not in st.session_state:
        st.session_state.paper_summaries = {}
    
    # Set up the paper sources
    sources = {
        "arXiv": ArxivSource(),
        "Semantic Scholar": SemanticScholarSource(),
        "Crossref": CrossrefSource()
    }
    
    # Search interface
    st.subheader("Search for papers")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Enter search terms", placeholder="e.g., solar panel tracking system")
    
    with col2:
        source_name = st.selectbox("Source", list(sources.keys()))
    
    # Option to upload a PDF directly
    st.subheader("Or upload a PDF directly")
    uploaded_file = st.file_uploader("Upload a research paper", type="pdf")
    
    if st.button("Search") and search_query:
        with st.spinner(f"Searching {source_name}..."):
            try:
                results = sources[source_name].search(search_query)
                if results:
                    st.session_state.search_results = results
                    st.session_state.current_source = source_name
                else:
                    st.warning(f"No results found for '{search_query}' in {source_name}.")
                    st.session_state.search_results = []
            except Exception as e:
                st.error(f"Error searching {source_name}: {str(e)}")
                st.session_state.search_results = []
    
    # Process uploaded PDF
    if uploaded_file:
        with st.spinner("Processing your uploaded PDF..."):
            try:
                pdf_content = uploaded_file.getvalue()
                summarizer = PaperSummarizer()
                summaries = summarizer.summarize_paper(pdf_content)
                
                paper_id = f"uploaded_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                paper = {
                    'id': paper_id,
                    'title': uploaded_file.name,
                    'authors': ['Unknown'],
                    'published': 'Unknown',
                    'source': 'upload',
                    'abstract': '',
                    'url': ''
                }
                
                st.session_state.paper_summaries[paper_id] = {
                    'paper': paper,
                    'summaries': summaries
                }
                st.success("Paper summarized successfully!")
            except Exception as e:
                st.error(f"Error processing uploaded PDF: {str(e)}")
    
    # Display search results
    if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
        st.subheader("Search Results")
        for i, paper in enumerate(st.session_state.search_results):
            with st.expander(f"{i+1}. {paper['title']}"):
                st.write(f"**Authors:** {', '.join(paper['authors'])}")
                st.write(f"**Published:** {paper['published']}")
                if paper['abstract']:
                    st.write(f"**Abstract:** {paper['abstract'][:300]}...")
                else:
                    st.write("**Abstract:** Not available")
                
                if st.button("Summarize This Paper", key=f"summarize_{i}"):
                    source = sources[st.session_state.current_source]
                    paper_id = paper['id']
                    
                    with st.spinner("Downloading and processing paper..."):
                        pdf_content = source.get_paper(paper_id)
                        
                        if pdf_content:
                            try:
                                summarizer = PaperSummarizer()
                                summaries = summarizer.summarize_paper(pdf_content)
                                st.session_state.paper_summaries[paper_id] = {
                                    'paper': paper,
                                    'summaries': summaries
                                }
                                st.success("Paper summarized successfully!")
                            except Exception as e:
                                st.error(f"Error summarizing paper: {str(e)}")
                        else:
                            st.error("Could not download the paper. It might be behind a paywall or not available in PDF format.")
    
    # Display summaries
    if st.session_state.paper_summaries:
        st.subheader("Paper Summaries")
        for paper_id, data in st.session_state.paper_summaries.items():
            paper = data['paper']
            summaries = data['summaries']
            
            with st.expander(f"Summary of: {paper['title']}"):
                for item in summaries:
                    section_title = item[0]
                    summary = item[1]
                    rouge_scores = item[2] if len(item) > 2 else None
                    
                    st.markdown(f"### {section_title}")
                    st.write(summary)
                    
                    # Display ROUGE scores if available - NO NESTED EXPANDERS
                    if rouge_scores:
                        st.markdown("##### Quality Metrics (ROUGE Scores)")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("ROUGE-1", f"{rouge_scores['rouge1']:.2f}")
                        with col2:
                            st.metric("ROUGE-2", f"{rouge_scores['rouge2']:.2f}")
                        with col3:
                            st.metric("ROUGE-L", f"{rouge_scores['rougeL']:.2f}")
                        
                        st.markdown("""
                        **ROUGE Score Interpretation:**
                        - **ROUGE-1**: Overlap of unigrams (single words)
                        - **ROUGE-2**: Overlap of bigrams (word pairs)
                        - **ROUGE-L**: Longest common subsequence
                        
                        Higher scores (closer to 1.0) indicate better summary quality compared to the source text.
                        """)
                
                if st.button("Save to My Library", key=f"save_{paper_id}"):
                    if "my_library" not in st.session_state:
                        st.session_state.my_library = []
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.my_library.append({
                        'paper': paper,
                        'summaries': summaries,
                        'added_on': timestamp
                    })
                    st.success("Added to your library!")
    
    # My Library
    if "my_library" in st.session_state and st.session_state.my_library:
        st.subheader("My Library")
        for i, item in enumerate(st.session_state.my_library):
            paper = item['paper']
            with st.expander(f"{paper['title']}"):
                st.write(f"**Authors:** {', '.join(paper['authors'])}")
                st.write(f"**Added on:** {item['added_on']}")
                st.write(f"**Source:** {paper['source']}")
                
                if st.button("View Summary", key=f"view_{i}"):
                    st.markdown("### Section Summaries")
                    for section_data in item['summaries']:
                        section_title = section_data[0]
                        summary = section_data[1]
                        rouge_scores = section_data[2] if len(section_data) > 2 else None
                        
                        st.markdown(f"#### {section_title}")
                        st.write(summary)
                        
                        # Display ROUGE scores if available - NO NESTED EXPANDERS
                        if rouge_scores:
                            st.markdown("##### Quality Metrics (ROUGE Scores)")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("ROUGE-1", f"{rouge_scores['rouge1']:.2f}")
                            with col2:
                                st.metric("ROUGE-2", f"{rouge_scores['rouge2']:.2f}")
                            with col3:
                                st.metric("ROUGE-L", f"{rouge_scores['rougeL']:.2f}")
                
                if st.button("Remove from Library", key=f"remove_{i}"):
                    st.session_state.my_library.pop(i)
                    st.experimental_rerun()