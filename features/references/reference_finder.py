import streamlit as st
import requests
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
import time
from datetime import datetime
import json

class ResearchPaperSearchAssistant:
    def __init__(self):
        self.platforms = {
            "Semantic Scholar": self._search_semantic_scholar,
            "arXiv": self._search_arxiv,
            "CrossRef": self._search_crossref
        }
        # Initialize SciBERT model and tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(self.device)
        self.papers_df = None
        self.embeddings = None
        
        # Evaluation metrics
        self.metrics = {
            'query_time': 0,
            'total_papers_found': 0,
            'papers_with_abstracts': 0,
            'avg_similarity_score': 0
        }

    def _search_semantic_scholar(self, query, limit=50):
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        headers = {'Content-Type': 'application/json'}
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,abstract,authors,year,url,citationCount,venue,referenceCount'
        }
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            papers_data = response.json().get('data', [])
            processed_papers = []
            for paper in papers_data:
                processed_papers.append({
                    'title': paper.get('title', 'No Title'),
                    'abstract': paper.get('abstract', 'No Abstract Available'),
                    'authors': [author.get('name', '') for author in paper.get('authors', [])],
                    'year': paper.get('year', 'Unknown'),
                    'url': paper.get('url', ''),
                    'platform': 'Semantic Scholar',
                    'citation_count': paper.get('citationCount', 0),
                    'venue': paper.get('venue', 'Unknown'),
                    'reference_count': paper.get('referenceCount', 0)
                })
            return processed_papers
        except Exception as e:
            st.error(f"Semantic Scholar API Error: {e}")
            return []

    def _search_arxiv(self, query, limit=50):
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': limit
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            # Basic XML parsing for arXiv results
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            # Define namespace
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            processed_papers = []
            for entry in root.findall('.//atom:entry', namespace):
                title = entry.find('./atom:title', namespace).text.strip()
                abstract = entry.find('./atom:summary', namespace).text.strip()
                
                # Extract authors
                authors = []
                for author in entry.findall('./atom:author/atom:name', namespace):
                    authors.append(author.text)
                
                # Extract URL
                url = ""
                for link in entry.findall('./atom:link', namespace):
                    if link.get('title') == 'pdf':
                        url = link.get('href')
                        break
                
                # Extract year from published date
                published = entry.find('./atom:published', namespace).text
                year = published.split('-')[0]
                
                processed_papers.append({
                    'title': title,
                    'abstract': abstract,
                    'authors': authors,
                    'year': year,
                    'url': url,
                    'platform': 'arXiv',
                    'citation_count': 1,  
                    'venue': 'arXiv',
                    'reference_count': 1
                })
            
            return processed_papers
        except Exception as e:
            st.error(f"arXiv API Error: {e}")
            return []

    def _search_crossref(self, query, limit=50):
        base_url = "https://api.crossref.org/works"
        params = {
            'query': query,
            'rows': limit
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            papers_data = response.json().get('message', {}).get('items', [])
            processed_papers = []
            for paper in papers_data:
                # Extract year safely
                year = 'Unknown'
                if paper.get('published'):
                    date_parts = paper.get('published', {}).get('date-parts', [['']])
                    if date_parts and date_parts[0]:
                        year = date_parts[0][0]
                
                processed_papers.append({
                    'title': paper.get('title', ['No Title'])[0] if isinstance(paper.get('title', []), list) else paper.get('title', 'No Title'),
                    'abstract': paper.get('abstract', 'No Abstract Available'),
                    'authors': [f"{author.get('given', '')} {author.get('family', '')}" for author in paper.get('author', [])],
                    'year': year,
                    'url': paper.get('URL', ''),
                    'platform': 'CrossRef',
                    'citation_count': paper.get('is-referenced-by-count', 0),
                    'venue': paper.get('container-title', ['Unknown'])[0] if isinstance(paper.get('container-title', []), list) else 'Unknown',
                    'reference_count': paper.get('references-count', 0)
                })
            return processed_papers
        except Exception as e:
            st.error(f"CrossRef API Error: {e}")
            return []

    def search_papers(self, query, platforms, start_year=None, end_year=None, limit=50):
        start_time = time.time()
        all_papers = []
        
        for platform in platforms:
            search_method = self.platforms.get(platform)
            if search_method:
                papers = search_method(query, limit)
                all_papers.extend(papers)
        
        # Filter by year if specified
        if start_year or end_year:
            filtered_papers = []
            current_year = datetime.now().year
            
            # Set defaults if not specified
            start_year = int(start_year) if start_year else 0
            end_year = int(end_year) if end_year else current_year
            
            for paper in all_papers:
                try:
                    paper_year = int(paper['year']) if paper['year'] != 'Unknown' else 0
                    if start_year <= paper_year <= end_year:
                        filtered_papers.append(paper)
                except (ValueError, TypeError):
                    # Skip papers with invalid year format
                    pass
            
            all_papers = filtered_papers
        
        # Update metrics
        self.metrics['query_time'] = time.time() - start_time
        self.metrics['total_papers_found'] = len(all_papers)
        self.metrics['papers_with_abstracts'] = sum(1 for paper in all_papers if paper['abstract'] != 'No Abstract Available')
        
        return all_papers

    def preprocess_text(self, text):
        if not text or text == 'No Abstract Available':
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text

    def get_scibert_embedding(self, text):
        # Preprocess and get embedding using SciBERT
        text = self.preprocess_text(text)
        if not text:
            # Return zero vector for empty text
            return np.zeros(768)
        
        # Tokenize and get embedding
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()[0]  

    def prepare_recommendation_system(self, papers_data):
        # Filter papers that have abstracts
        filtered_papers = [
            paper for paper in papers_data
            if paper.get('abstract', '') and paper.get('abstract', '') != 'No Abstract Available'
        ]
        
        self.papers_df = pd.DataFrame(filtered_papers)
        
        if len(self.papers_df) == 0:
            st.warning("No papers with abstracts found. Cannot prepare recommendation system.")
            return
        
        # Get SciBERT embeddings for all papers
        st.info(f"Computing SciBERT embeddings for {len(self.papers_df)} papers...")
        
        # Process in batches to show progress
        embeddings = []
        batch_size = 5  # Process 5 papers at a time to show progress
        total_batches = (len(self.papers_df) + batch_size - 1) // batch_size
        
        progress_bar = st.progress(0)
        for i in range(0, len(self.papers_df), batch_size):
            batch = self.papers_df.iloc[i:i+batch_size]
            batch_embeddings = [self.get_scibert_embedding(abstract) for abstract in batch['abstract']]
            embeddings.extend(batch_embeddings)
            
            # Update progress
            progress = min((i + batch_size) / len(self.papers_df), 1.0)
            progress_bar.progress(progress)
        
        self.embeddings = np.array(embeddings)
        progress_bar.empty()
        
        st.success(f"Recommendation system prepared with {len(self.papers_df)} papers")

    def recommend_papers(self, topic, top_k=10, min_year=None, max_year=None):
        if self.embeddings is None or len(self.embeddings) == 0:
            st.error("Recommendation system not prepared.")
            return pd.DataFrame()
        
        # Get embedding for the query topic
        query_embedding = self.get_scibert_embedding(topic)
        
        # Compute cosine similarity
        cosine_similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Create recommendations dataframe
        recommendations_df = self.papers_df.copy()
        recommendations_df['similarity_score'] = cosine_similarities
        
        # Filter by year if specified
        if min_year and max_year:
            min_year = int(min_year)
            max_year = int(max_year)
            
            year_filtered = recommendations_df[
                recommendations_df['year'].apply(
                    lambda y: min_year <= int(y) <= max_year if y != 'Unknown' else False
                )
            ]
            
            # If filtered results exist, use them; otherwise, use all results
            if not year_filtered.empty:
                recommendations_df = year_filtered
        
        # Sort by similarity score and get top_k
        top_recommendations = recommendations_df.sort_values(
            'similarity_score', ascending=False
        ).head(top_k)
        
        # Update metrics
        self.metrics['avg_similarity_score'] = top_recommendations['similarity_score'].mean()
        
        return top_recommendations[['title', 'abstract', 'authors', 'year', 'url', 'platform', 'citation_count', 'venue', 'reference_count', 'similarity_score']]

    def get_evaluation_metrics(self):
        return self.metrics

    def calculate_impact_score(self, paper):
        """Calculate a paper's impact score based on citations, recency, and venue"""
        citation_count = paper.get('citation_count', 1)
        reference_count = paper.get('reference_count', 1)
        similarity = paper.get('similarity_score', 0)
        
        # Calculate recency factor (papers from last 5 years get a boost)
        current_year = datetime.now().year
        paper_year = paper.get('year', 'Unknown')
        
        if paper_year != 'Unknown':
            try:
                year_factor = max(0, 1 - (current_year - int(paper_year)) / 10)
            except (ValueError, TypeError):
                year_factor = 0
        else:
            year_factor = 0
        
        # Final impact score formula
        impact_score = (0.5 * similarity) + (0.3 * min(citation_count / 100, 1)) + (0.1 * year_factor) + (0.1 * min(reference_count / 50, 1))
        
        return impact_score

def run_references():
    st.subheader("🔬 Research Reference Papers")
    
    # Initialize the research assistant
    if 'research_assistant' not in st.session_state:
        st.session_state.research_assistant = ResearchPaperSearchAssistant()
    
    research_assistant = st.session_state.research_assistant
    
    # Default platforms to search
    available_platforms = list(research_assistant.platforms.keys())

    # Main search interface (simplified)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        research_topic = st.text_input("Enter Research Topic", "machine learning in healthcare")
    
    with col2:
        top_k = st.slider("Number of Papers", min_value=3, max_value=15, value=5)
    
    # Year range slider
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=1990, max_value=current_year, value=current_year-5)
    with col2:
        end_year = st.number_input("End Year", min_value=start_year, max_value=current_year, value=current_year)
    
    # Hidden options in expandable section for advanced users
    with st.expander("Advanced Options"):
        selected_platforms = st.multiselect(
            "Select Research Platforms", available_platforms, default=available_platforms
        )
    
    # Use all platforms by default if none selected
    if not selected_platforms:
        selected_platforms = available_platforms
    
    search_button = st.button("Find Research Papers")

    # Search and recommend
    if search_button and research_topic:
        with st.spinner("Searching for papers..."):
            papers = research_assistant.search_papers(
                research_topic, selected_platforms, start_year, end_year, limit=50
            )
            
            if not papers:
                st.warning("No papers found. Try adjusting your search parameters.")
                return
                
        with st.spinner("Processing papers with SciBERT..."):
            research_assistant.prepare_recommendation_system(papers)
            
        with st.spinner("Finding most relevant papers..."):
            recommendations = research_assistant.recommend_papers(
                research_topic, top_k=top_k, min_year=start_year, max_year=end_year
            )

        # Get evaluation metrics
        metrics = research_assistant.get_evaluation_metrics()
        
        # Display metrics in collapsible section
        with st.expander("Search Metrics"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Papers Found", metrics['total_papers_found'])
            col2.metric("Query Time", f"{metrics['query_time']:.2f}s")
            col3.metric("Avg. Similarity", f"{metrics['avg_similarity_score']:.2f}")

        # Display recommendations
        if not recommendations.empty:
            st.header("Top Research Papers")
            
            # Calculate impact scores for each paper
            impact_scores = []
            for _, paper in recommendations.iterrows():
                paper_dict = paper.to_dict()
                impact_score = research_assistant.calculate_impact_score(paper_dict)
                impact_scores.append(impact_score)
            
            recommendations['impact_score'] = impact_scores
            
            # Display papers with their scores
            for idx, paper in recommendations.iterrows():
                with st.expander(f"{paper['title']} ({paper['year']})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                        st.markdown(f"**Venue:** {paper['venue']}")
                        st.markdown(f"**Abstract:** {paper['abstract']}")
                        st.markdown(f"**URL:** [Link]({paper['url']})")
                    
                    with col2:
                        # Display relevance scores using meter-style indicators
                        st.markdown("### Relevance Metrics")
                        st.markdown(f"**Platform:** {paper['platform']}")
                        st.markdown(f"**Citations:** {paper['citation_count']}")
                        st.markdown(f"**References:** {paper['reference_count']}")
                        
                        # Visual indicators for similarity and impact
                        st.markdown("**Semantic Relevance:**")
                        st.progress(float(min(paper['similarity_score'], 1.0)))
                        st.caption(f"{paper['similarity_score']:.2f}")
                        
                        st.markdown("**Impact Score:**")
                        st.progress(float(min(paper['impact_score'], 1.0)))
                        st.caption(f"{paper['impact_score']:.2f}")
        else:
            st.warning("No recommendations found. Try adjusting search parameters.")

    # Documentation to explain evaluation metrics
    with st.expander("How are papers evaluated?"):
        st.markdown("""
        ### Paper Recommendation Criteria
        
        Our system evaluates papers based on multiple factors:
        
        1. **Semantic Relevance (50%)**: Uses SciBERT language model to understand the semantic meaning of your research topic and paper abstracts
        
        2. **Citation Impact (30%)**: Papers with more citations have greater influence in their field
        
        3. **Recency (10%)**: Newer papers get a higher score to ensure you get current research
        
        4. **Reference Count (10%)**: Papers that reference more sources often provide better literature reviews
        
        The system uses these weighted factors to create an overall Impact Score that helps identify the most valuable papers for your research.
        """)