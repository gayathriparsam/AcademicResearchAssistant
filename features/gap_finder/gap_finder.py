import streamlit as st
import requests
import xml.etree.ElementTree as ET
import numpy as np
import re
import pandas as pd
import plotly.express as px
from collections import Counter
import time
from datetime import datetime
import json
from keybert import KeyBERT
import torch
from transformers import AutoTokenizer, AutoModel

# Function to load SciBERT model
@st.cache_resource
def load_scibert_model():
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    return tokenizer, model

# Function to get SciBERT embeddings
def get_scibert_embeddings(texts, tokenizer, model):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            # Tokenize and get model output
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
            outputs = model(**inputs)
            
            # Mean Pooling - Take average of all token embeddings
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze().numpy()
            
            embeddings.append(embedding)
    
    return np.array(embeddings)

def fetch_papers(query, limit=75):
    """Fetch papers from Semantic Scholar, arXiv, and CrossRef with improved error handling and year filtering."""
    papers = []
    limit_per_source = limit // 3
    current_year = datetime.now().year
    
    # Progress bar for paper fetching
    paper_progress = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Fetching papers from multiple sources...")
    
    # Semantic Scholar
    ss_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    ss_params = {'query': query, 'limit': limit_per_source * 2, 'fields': 'title,abstract,year,authors'}  # Get more to filter
    try:
        response = requests.get(ss_url, params=ss_params, timeout=10)
        if response.status_code == 200:
            data = response.json().get('data', [])
            for p in data:
                if p.get('abstract') and p.get('year') and int(p.get('year', 0)) >= (current_year - 3):
                    title = p.get('title', 'Untitled')
                    abstract = p.get('abstract', '')
                    year = p.get('year', 'Unknown')
                    authors = ", ".join([a.get('name', '') for a in p.get('authors', [])[:3]])
                    if len(p.get('authors', [])) > 3:
                        authors += " et al."
                    papers.append({
                        'title': title,
                        'abstract': abstract,
                        'source': 'Semantic Scholar', 
                        'year': year,
                        'authors': authors
                    })
        else:
            st.warning(f"Semantic Scholar returned status code {response.status_code}")
    except Exception as e:
        st.warning(f"Semantic Scholar's acting up—moving on! Error: {str(e)}")
    
    paper_progress.progress(33)
    progress_text.text("Fetching papers from arXiv...")
    
    # arXiv with year filtering
    arxiv_url = "http://export.arxiv.org/api/query"
    arxiv_params = {'search_query': f'all:{query} AND submittedDate:[{current_year-3} TO {current_year}]', 
                   'max_results': limit_per_source * 2}
    try:
        response = requests.get(arxiv_url, params=arxiv_params, timeout=10)
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            for entry in entries:
                abstract = entry.findtext('{http://www.w3.org/2005/Atom}summary', '')
                if abstract:
                    title = entry.findtext('{http://www.w3.org/2005/Atom}title', 'Untitled')
                    published = entry.findtext('{http://www.w3.org/2005/Atom}published', '')
                    year = published[:4] if published else 'Unknown'
                    if year != 'Unknown' and int(year) >= (current_year - 3):
                        authors = ", ".join([author.findtext('{http://www.w3.org/2005/Atom}name', '') 
                                           for author in entry.findall('.//{http://www.w3.org/2005/Atom}author')[:3]])
                        if len(entry.findall('.//{http://www.w3.org/2005/Atom}author')) > 3:
                            authors += " et al."
                        papers.append({
                            'title': title,
                            'abstract': abstract,
                            'source': 'arXiv',
                            'year': year,
                            'authors': authors
                        })
        else:
            st.warning(f"arXiv returned status code {response.status_code}")
    except Exception as e:
        st.warning(f"arXiv's being shy—skipping it! Error: {str(e)}")
    
    paper_progress.progress(66)
    progress_text.text("Fetching papers from CrossRef...")
    
    # CrossRef with year filter
    cr_url = "https://api.crossref.org/works"
    year_filter = f"from-pub-date:{current_year-3}"
    cr_params = {'query': query, 'rows': limit_per_source * 2, 'filter': year_filter}
    try:
        response = requests.get(cr_url, params=cr_params, timeout=10)
        if response.status_code == 200:
            items = response.json().get('message', {}).get('items', [])
            for item in items:
                abstract = re.sub(r'jats:p|<[^>]+>', '', item.get('abstract', '')) if item.get('abstract') else ''
                if abstract:
                    title = item.get('title', ['Untitled'])[0] if isinstance(item.get('title', []), list) else 'Untitled'
                    year = item.get('published-print', {}).get('date-parts', [['']])[0][0]
                    if not year:
                        year = item.get('published-online', {}).get('date-parts', [['']])[0][0]
                    year = year or 'Unknown'
                    
                    if year != 'Unknown' and int(year) >= (current_year - 3):
                        authors_list = item.get('author', [])
                        authors = ", ".join([f"{a.get('given', '')} {a.get('family', '')}" for a in authors_list[:3]])
                        if len(authors_list) > 3:
                            authors += " et al."
                            
                        papers.append({
                            'title': title,
                            'abstract': abstract,
                            'source': 'CrossRef',
                            'year': year,
                            'authors': authors
                        })
        else:
            st.warning(f"CrossRef returned status code {response.status_code}")
    except Exception as e:
        st.warning(f"CrossRef's out—sticking with what we've got! Error: {str(e)}")
    
    paper_progress.progress(100)
    progress_text.empty()
    paper_progress.empty()
    
    return papers[:limit]

def extract_keywords_keybert(abstracts, top_n=20):
    """Extract keywords using KeyBERT."""
    with st.spinner("Extracting keywords with KeyBERT..."):
        # Combine abstracts into one large text
        combined_text = " ".join(abstracts)
        
        # Load KeyBERT
        kw_model = KeyBERT()
        
        # Extract keywords
        keywords = kw_model.extract_keywords(
            combined_text, 
            keyphrase_ngram_range=(1, 2),  # Allow single words and bigrams
            stop_words='english',
            use_mmr=True,  # Use Maximal Marginal Relevance for diversity
            diversity=0.5,
            top_n=top_n
        )
        
        # Convert to dictionary with scores
        keyword_dict = {keyword: score for keyword, score in keywords}
        
        # Sort by score
        sorted_keywords = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_keywords

def simple_dimensionality_reduction(embeddings, n_components=2):
    """A simple PCA implementation without using sklearn."""
    # Center the data
    mean = np.mean(embeddings, axis=0)
    centered_data = embeddings - mean
    
    # Compute covariance matrix
    cov_matrix = np.cov(centered_data, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Take the first n_components eigenvectors
    principal_components = eigenvectors[:, :n_components]
    
    # Project the data
    projected_data = np.dot(centered_data, principal_components)
    
    return projected_data

def find_gaps(papers, tokenizer, model, similarity_threshold=0.75, visualization=True):
    """Use SciBERT to find research gaps with visualizations."""
    if not papers:
        return [], None
    
    with st.spinner("Analyzing paper embeddings with SciBERT..."):
        abstracts = [p['abstract'] for p in papers]
        
        # Generate embeddings using SciBERT
        embeddings = get_scibert_embeddings(abstracts, tokenizer, model)
        
        # Find the center of the field
        field_center = np.mean(embeddings, axis=0)
        
        # Calculate similarities to the center
        similarities = np.dot(embeddings, field_center) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(field_center)
        )
        
        # Find outliers
        outlier_indices = [i for i, s in enumerate(similarities) if s < similarity_threshold]
        
        # If no clear outliers, take the most dissimilar papers
        if len(outlier_indices) < 3:
            outlier_indices = similarities.argsort()[:5]
        
        # Add similarity scores to papers
        for i, paper in enumerate(papers):
            paper['similarity'] = float(similarities[i])
        
        # Prepare gap data
        gap_data = [papers[i] for i in outlier_indices]
        
        # Create visualization if requested
        viz_fig = None
        if visualization and len(embeddings) > 5:
            # Use our simple PCA function 
            reduced_embeddings = simple_dimensionality_reduction(embeddings, n_components=2)
            
            # Create dataframe for plotting
            df = pd.DataFrame({
                'x': reduced_embeddings[:, 0],
                'y': reduced_embeddings[:, 1],
                'title': [p['title'] for p in papers],
                'source': [p['source'] for p in papers],
                'year': [p['year'] for p in papers],
                'similarity': similarities,
                'is_gap': [i in outlier_indices for i in range(len(papers))]
            })
            
            # Create interactive scatter plot
            viz_fig = px.scatter(
                df, x='x', y='y', 
                color='similarity', size=(1-df['similarity'])*10+5,
                hover_data=['title', 'source', 'year'],
                labels={'similarity': 'Similarity to field center'},
                color_continuous_scale='Viridis',
                title='Research Landscape: Potential Gaps in Dark Blue'
            )
            
            # Highlight potential gap papers
            gap_df = df[df['is_gap']]
            gap_trace = px.scatter(
                gap_df, x='x', y='y',
                text=[f"Gap {i+1}" for i in range(len(gap_df))],
                color_discrete_sequence=['red']
            ).data[0]
            
            viz_fig.add_trace(gap_trace)
            viz_fig.update_traces(marker=dict(line=dict(width=2, color='DarkRed')),
                                 selector=dict(mode='markers+text'))
            
            # Improve layout
            viz_fig.update_layout(
                height=500,
                legend_title_text='Paper Data',
                xaxis_title="First Principal Component",
                yaxis_title="Second Principal Component"
            )
        
        return gap_data, viz_fig

def analyze_keyword_coverage(papers, gap_papers):
    """Analyze keyword coverage to find potential research gaps using KeyBERT."""
    all_abstracts = [p['abstract'] for p in papers]
    gap_abstracts = [p['abstract'] for p in gap_papers]
    
    # Extract keywords from all papers and gap papers using KeyBERT
    all_keywords = dict(extract_keywords_keybert(all_abstracts, top_n=30))
    gap_keywords = dict(extract_keywords_keybert(gap_abstracts, top_n=20))
    
    # Find keywords that are more prominent in gap papers (potential new directions)
    keyword_opportunities = {}
    for keyword, score in gap_keywords.items():
        if keyword in all_keywords:
            # Calculate relative importance in gap papers vs all papers
            gap_importance = score
            all_importance = all_keywords[keyword]
            
            # If keyword is more important in gap papers, it might represent an opportunity
            if gap_importance > all_importance * 1.2:
                keyword_opportunities[keyword] = gap_importance / all_importance
        else:
            # Keywords unique to gap papers are very interesting
            keyword_opportunities[keyword] = 5.0  # Arbitrary high score
    
    # Sort keywords by opportunity score
    sorted_opportunities = sorted(keyword_opportunities.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_opportunities[:10]  # Return top 10 opportunity keywords

def get_mistral_response(prompt, system_prompt="", api_url="http://localhost:11434/api/generate"):
    """Get response from locally running Mistral model via Ollama API"""
    
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "mistral",
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            st.warning(f"Error from Mistral API: {response.status_code}")
            return "I couldn't generate gap ideas at this time. Please check your Ollama installation."
    except Exception as e:
        st.warning(f"Error connecting to Ollama API: {e}")
        return "I couldn't connect to the Mistral model. Please make sure Ollama is running."

def generate_gap_ideas_with_mistral(paper, topic, opportunity_keywords):
    """Generate research gap ideas using Mistral LLM"""
    
    # Create a system prompt for research gap analysis
    system_prompt = """You are a research assistant specialized in identifying research gaps and opportunities. 
    Analyze the provided research paper information and suggest novel research directions and gaps that could be filled.
    Be specific, innovative, and focus on actionable research ideas."""
    
    # Create the main prompt with paper info and opportunity keywords
    prompt = f"""
    TOPIC: {topic}
    
    PAPER TITLE: {paper['title']}
    PAPER YEAR: {paper['year']}
    PAPER AUTHORS: {paper['authors']}
    PAPER ABSTRACT: {paper['abstract']}
    
    EMERGING KEYWORDS IN THE FIELD: {', '.join([k for k, _ in opportunity_keywords[:5]])}
    
    Based on this paper and the trending keywords, identify 1-2 specific research gaps or opportunities. 
    Consider:
    1. How this paper differs from mainstream research
    2. What questions it raises but doesn't answer
    3. How the emerging keywords could be integrated with this paper's approach
    4. What methodological innovations could be applied
    
    Provide concise, specific research gap suggestions (max 3 sentences each).
    """
    
    # Get response from Mistral
    with st.spinner("Generating research gap ideas with Mistral LLM..."):
        response = get_mistral_response(prompt, system_prompt)
    
    # Clean up response if needed
    response = response.strip()
    
    # Fallback if Mistral fails
    if not response or "couldn't" in response:
        return f"This paper from {paper['year']} differs from mainstream research on {topic.split(':')[0]}. Consider how its methodologies could be applied to solve current challenges in light of emerging keywords like {', '.join([k for k, _ in opportunity_keywords[:2]])}."
    
    return response

def run_gap_finder():
    st.title("🕳️ Research Gap Finder")
    st.write("Discover untapped research opportunities and emerging trends in your field")
    
    # Load SciBERT model
    with st.spinner("Loading SciBERT model..."):
        tokenizer, model = load_scibert_model()
    
    # Set default parameters - no user input required for these
    similarity_threshold = 0.75
    paper_limit = 75
    show_visualization = True
    
    # Main input area - only topic required
    topic = st.text_input("Research Topic", 
                         "Global AI Governance in Healthcare: A Cross-Jurisdictional Regulatory Analysis",
                         help="Enter a specific research topic to find gaps")
    
    search_button = st.button("Find Research Gaps", type="primary", use_container_width=True)
    
    if search_button:
        # Step 1: Fetch papers
        with st.spinner("Searching for recent papers (past 3 years)..."):
            papers = fetch_papers(topic, limit=paper_limit)
            
            if not papers:
                st.error("No papers found. Try a broader topic or check your internet connection.")
                return
            
            st.success(f"Found {len(papers)} papers from the past 3 years related to your topic")
        
        # Step 2: Analyze gaps with SciBERT
        gaps, viz_fig = find_gaps(papers, tokenizer, model, similarity_threshold, show_visualization)
        
        # Show visualization if available
        if viz_fig and show_visualization:
            st.plotly_chart(viz_fig, use_container_width=True)
            st.caption("Papers further from the center (darker blue) represent potential research gaps")
        
        # Step 3: Advanced keyword analysis with KeyBERT
        opportunity_keywords = analyze_keyword_coverage(papers, gaps)
        
        # Show keyword opportunities
        st.subheader("🔍 Emerging Research Keywords")
        keyword_df = pd.DataFrame(opportunity_keywords, columns=["Keyword", "Opportunity Score"])
        keyword_df["Opportunity Score"] = keyword_df["Opportunity Score"].round(2)
        
        # Create horizontal bar chart for keywords
        keyword_fig = px.bar(
            keyword_df, 
            y="Keyword", 
            x="Opportunity Score", 
            orientation='h',
            title="Keywords More Common in Gap Papers",
            color="Opportunity Score",
            color_continuous_scale="Viridis",
        )
        
        st.plotly_chart(keyword_fig, use_container_width=True)
        
        # Step 4: Display gap opportunities with Mistral-generated ideas
        st.subheader("🌟 High-Potential Research Gap Opportunities")
        st.write("These papers stand out from the mainstream literature and may indicate research gaps")
        
        with st.expander("What makes these good gap opportunities?", expanded=False):
            st.markdown("""
            Papers are ranked as potential gap opportunities when they:
            - Differ significantly from the central themes in the field
            - Use unusual approaches or methodologies
            - Address similar problems from different angles
            - Contain keywords that aren't common in the mainstream literature
            
            Not every outlier represents a viable research gap, but they often point to fruitful directions for original research.
            """)
        
        # Display gap papers in cards using columns
        for i, paper in enumerate(gaps[:5], 1):
            st.markdown(f"### Gap Opportunity {i}: {paper['title']}")
            
            col1, col2 = st.columns([7, 3])
            with col1:
                st.markdown(f"**Source**: {paper['source']} ({paper['year']})")
                st.markdown(f"**Authors**: {paper['authors']}")
                st.markdown(f"**Similarity Score**: {paper['similarity']:.2f}")
                
                # Show abstract in expandable section
                with st.expander("Abstract", expanded=False):
                    st.write(paper['abstract'])
                
            with col2:
                # Research gap idea generated by Mistral
                st.markdown("#### Research Gap Idea")
                idea = generate_gap_ideas_with_mistral(paper, topic, opportunity_keywords)
                st.info(idea)
            
            st.divider()
            
        # Step 5: Final summary and tips
        st.subheader("✨ Research Strategy Tips")
        st.markdown("""
        To leverage these gaps effectively:
        1. **Explore outlier combinations**: Consider how elements from different gap papers might be combined
        2. **Cross-domain application**: Look for methods used in other fields that could be applied to your topic
        3. **Investigate emerging keywords**: Focus on the trending keywords identified in the analysis
        4. **Challenge assumptions**: Question why the gap papers differ from the mainstream approach
        """)
        
        # Prepare data for download
        download_data = []
        for paper in papers:
            download_data.append({
                'Title': paper['title'],
                'Source': paper['source'],
                'Year': paper['year'],
                'Authors': paper['authors'],
                'Similarity': paper.get('similarity', 'N/A'),
                'Is Gap Paper': paper in gaps,
                'Abstract': paper['abstract']
            })
        
        download_df = pd.DataFrame(download_data)
        csv = download_df.to_csv(index=False)
        
        st.download_button(
            label="Download All Paper Data (CSV)",
            data=csv,
            file_name=f"research_gaps_{topic.split(':')[0]}.csv",
            mime="text/csv",
        )









