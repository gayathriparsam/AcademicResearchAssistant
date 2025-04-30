import streamlit as st
import google.generativeai as genai
import json

# Configure Gemini API and list available models
def configure_genai_and_list_models(api_key: str):
    genai.configure(api_key=api_key)
    try:
        available_models = genai.list_models()
        model_names = [model.name for model in available_models if "generateContent" in model.supported_generation_methods]
        return model_names
    except Exception as e:
        raise Exception(f"Error listing models: {str(e)}")

# Initialize session state
def init_session_state():
    if 'current_section' not in st.session_state:
        st.session_state.current_section = 'Topic Selection'
    if 'paper_content' not in st.session_state:
        st.session_state.paper_content = {
            'topic': '',
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'discussion': '',
            'conclusion': '',
            'references': ''
        }

RESEARCH_GUIDELINES = {
    'abstract': """
    Guidelines for Abstract:
    - Keep it short and simple (200-250 words)
    - Include research problem, methodology, key findings
    - Use past tense for completed actions
    - Avoid citations and abbreviations
    """,
    'introduction': """
    Guidelines for Introduction:
    - Start with broader context
    - Narrow down to research problem
    - State research objectives clearly
    - Review relevant literature
    - Present research questions/hypotheses
    """,
    'methodology': """
    Guidelines for Methodology:
    - Describe the research design and approach
    - Explain data collection methods and tools
    - Detail the analysis techniques used
    - Justify choices with brief rationale
    - Use clear, step-by-step language
    """,
    'results': """
    Guidelines for Results:
    - Present findings in a logical order
    - Use tables, figures, or text as needed
    - Report data objectively without interpretation
    - Highlight key results relevant to objectives
    - Keep it concise and factual
    """,
    'discussion': """
    Guidelines for Discussion:
    - Interpret results in context of objectives
    - Compare findings with existing literature
    - Discuss implications and significance
    - Address limitations of the study
    - Suggest future research directions
    """,
    'conclusion': """
    Guidelines for Conclusion:
    - Summarize key findings and contributions
    - Restate the research problem and solution
    - Avoid introducing new data or ideas
    - Keep it brief and impactful
    - End with a strong closing statement
    """,
    'references': """
    Guidelines for References:
    - List all cited sources accurately
    - Follow a consistent citation style (e.g., APA, MLA)
    - Include full bibliographic details
    - Arrange alphabetically or by order of appearance
    - Double-check for completeness and formatting
    """
}

def generate_section_guidance(model, section: str, topic: str) -> str:
    prompt = f"""
    As a research paper writing assistant, provide detailed guidance for writing the {section} section 
    of a research paper on the topic: {topic}
    
    Include:
    1. Specific points to cover
    2. Common mistakes to avoid
    3. Writing style recommendations
    4. Section-specific tips
    5. Examples or templates where appropriate
    
    Base your response on standard academic writing practices and these guidelines:
    {RESEARCH_GUIDELINES.get(section.lower(), '')}
    """
    response = model.generate_content(prompt)
    return response.text

def run_writing():
    st.subheader("✍️ Guide Research Writing")

    # API Key input in sidebar
    with st.sidebar:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if not api_key:
            st.warning("Please enter your Gemini API key to continue")
            return

    # Configure Gemini and select model
    try:
        model_names = configure_genai_and_list_models(api_key)
        if not model_names:
            st.error("No models available for content generation with your API key.")
            return
        with st.sidebar:
            selected_model_name = st.selectbox("Select Gemini Model", model_names, index=0)
        model = genai.GenerativeModel(selected_model_name)
        st.sidebar.success(f"Using model: {selected_model_name}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check your API key and make sure you have access to Gemini models.")
        return

    init_session_state()

    # Section navigation in sidebar
    sections = [
        "Topic Selection", "Abstract", "Introduction", "Methodology",
        "Results", "Discussion", "Conclusion", "References"
    ]
    with st.sidebar:
        selected_section = st.radio("Navigate Sections", sections)
    st.session_state.current_section = selected_section

    # Topic Selection
    if selected_section == "Topic Selection":
        st.write("### Research Topic Selection")
        topic = st.text_input("Enter your research topic:", 
                             value=st.session_state.paper_content['topic'])
        if st.button("Get Topic Feedback") and topic:
            try:
                with st.spinner("Analyzing topic..."):
                    prompt = f"""
                    Analyze this research topic: {topic}
                    
                    Provide:
                    1. Topic strength evaluation
                    2. Suggested refinements
                    3. Potential research questions
                    4. Key areas to focus on
                    5. Possible challenges
                    """
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                st.session_state.paper_content['topic'] = topic
            except Exception as e:
                st.error(f"Error generating content: {str(e)}")

    # Other Sections
    else:
        if not st.session_state.paper_content['topic']:
            st.warning("Please select a topic first")
            return
        st.write(f"### {selected_section}")
        
        # Display section guidelines
        with st.expander("View Section Guidelines", expanded=True):
            try:
                guidance = generate_section_guidance(
                    model, selected_section, st.session_state.paper_content['topic']
                )
                st.markdown(guidance)
            except Exception as e:
                st.error(f"Error generating guidelines: {str(e)}")

        # Section content input
        section_key = selected_section.lower()
        section_content = st.text_area(
            f"Write your {selected_section} here:",
            value=st.session_state.paper_content[section_key],
            height=300
        )
        
        if st.button("Get Feedback"):
            try:
                with st.spinner("Analyzing content..."):
                    prompt = f"""
                    Review this {selected_section} section for a research paper on 
                    {st.session_state.paper_content['topic']}.
                    
                    Content to review:
                    {section_content}
                    
                    Provide:
                    1. Content evaluation
                    2. Style and clarity feedback
                    3. Specific improvement suggestions
                    4. Missing elements
                    5. Strengths of the current version
                    """
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                st.session_state.paper_content[section_key] = section_content
            except Exception as e:
                st.error(f"Error generating feedback: {str(e)}")

    # Export functionality in sidebar
    with st.sidebar:
        if st.button("Export Paper Content"):
            paper_content = json.dumps(st.session_state.paper_content, indent=2)
            st.download_button(
                label="Download Paper Content",
                data=paper_content,
                file_name="research_paper_content.json",
                mime="application/json"
            )