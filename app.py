# -*- coding: utf-8 -*-
"""
Simple ATS Resume Analyzer
Upload resume and job description to get match score
"""

import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding='utf-8')
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding='utf-8')

from pdf_extractor import PDFExtractor
from ats_pipeline import ATSPipeline
from rag_skills_extractor import RAGSkillsExtractor
from llm_extractor import LLMResumeExtractor
from job_role_predictor import JobRolePredictor
import json

# Get the directory where app.py is located
APP_DIR = Path(__file__).parent

# Page config
st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    pdf_extractor = PDFExtractor()
    ats_pipeline = ATSPipeline(use_spacy=True)
    rag_extractor = RAGSkillsExtractor(
        skills_csv_path=str(APP_DIR / "data" / "skills_exploded (2).csv"),
        max_skills=10000
    )
    try:
        job_predictor = JobRolePredictor(
            model_path=str(APP_DIR / "JobPrediction_Model")
        )
    except Exception as e:
        st.warning(f"Job predictor not available: {e}")
        job_predictor = None
    return pdf_extractor, ats_pipeline, rag_extractor, job_predictor

pdf_extractor, ats_pipeline, rag_extractor, job_predictor = init_components()

# Main app
def main():
    st.title("📄 ATS Resume Analyzer")
    st.markdown("### Analyze how well your resume matches a job description")
    
    # Two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Resume")
        resume_file = st.file_uploader("Upload Resume (PDF)", type=['pdf'], key="resume")
        
    with col2:
        st.subheader("💼 Job Description")
        job_description = st.text_area(
            "Paste Job Description",
            height=300,
            placeholder="Paste the complete job description here..."
        )
    
    # Analysis options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_rag = st.checkbox("Enable RAG Skills Extraction", value=True, 
                             help="Use semantic similarity to detect 1000+ skills")
    with col2:
        use_llm = st.checkbox("Enable LLM Analysis", value=False,
                             help="Use Gemini AI for structured extraction (slower)")
    with col3:
        if use_llm:
            gemini_api_key = st.text_input("Gemini API Key", type="password",
                                          value="")
    
    # Analyze button
    if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
        if not resume_file:
            st.error("Please upload a resume PDF")
            return
        
        if not job_description.strip():
            st.error("Please paste a job description")
            return
        
        # Save resume temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(resume_file.getbuffer())
            resume_path = tmp_file.name
        
        try:
            with st.spinner("Analyzing resume..."):
                # Extract resume text
                resume_text = pdf_extractor.extract_text_safe(resume_path)
                
                if not resume_text:
                    st.error("Could not extract text from PDF. Please ensure it's a text-based PDF, not a scanned image.")
                    return
                
                st.success(f"✅ Extracted {len(resume_text)} characters from resume")
                
                # Analyze format first
                st.info("Analyzing resume format...")
                format_analysis = pdf_extractor.detect_cv_structure(resume_text)
                
                # Run ATS analysis
                st.info("Running ATS analysis...")
                ats_results = ats_pipeline.analyze(
                    resume_path,
                    job_description,
                    verbose=False,
                    analyze_format=False
                )
                
                # Add format analysis to results
                ats_results['format_analysis'] = format_analysis
                
                # RAG analysis
                rag_skills = []
                if use_rag:
                    st.info("Running RAG skills extraction...")
                    rag_skills = rag_extractor.extract_skills_rag(resume_text, threshold=0.65)
                
                # LLM analysis
                llm_analysis = None
                if use_llm and gemini_api_key:
                    st.info("Running LLM analysis...")
                    llm_extractor = LLMResumeExtractor(
                        provider='gemini',
                        model='gemini-2.5-flash',
                        api_key=gemini_api_key
                    )
                    llm_analysis = llm_extractor.extract_from_text(resume_text)
                
                # Display results
                display_results(ats_results, rag_skills, llm_analysis, resume_text, job_predictor)
                
        finally:
            # Cleanup
            if os.path.exists(resume_path):
                os.unlink(resume_path)

def display_results(ats_results, rag_skills, llm_analysis, resume_text, job_predictor=None):
    """Display analysis results"""
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Overall Score
    overall_match = ats_results['similarity_scores']['overall_percentage']
    skills_match = ats_results['similarity_scores']['detailed_scores']['skills_match_rate'] * 100
    
    # Job prediction (if available)
    job_prediction = None
    if job_predictor and rag_skills:
        try:
            skills_text = ' '.join(rag_skills)
            job_prediction = job_predictor.predict_job_role(skills_text, top_n=5)
        except Exception as e:
            st.warning(f"Job prediction failed: {e}")
    
    # Big score display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score_color = "🟢" if overall_match >= 70 else "🟡" if overall_match >= 50 else "🔴"
        st.metric("Overall Match", f"{overall_match:.1f}%", delta=None)
        st.markdown(f"### {score_color} {get_match_level(overall_match)}")
    
    with col2:
        st.metric("Skills Match", f"{skills_match:.1f}%")
        st.metric("ATS Score", f"{ats_results['format_analysis']['ats_friendly_score']}/100")
    
    with col3:
        if rag_skills:
            st.metric("Skills Detected (RAG)", len(rag_skills))
        if llm_analysis:
            st.metric("Experience", f"{llm_analysis.get('total_experience_years', 0)} years")
        if job_prediction:
            st.metric("Predicted Role", job_prediction['predicted_role'].replace(' ', '\n'), 
                     delta=f"{job_prediction['confidence']:.0%} confidence")
    
    # Detailed Analysis Tabs
    tabs = [
        "🎯 Skills Analysis", 
        "📈 Scoring Breakdown",
        "🔍 RAG Skills",
        "🤖 LLM Analysis",
        "📄 Resume Format"
    ]
    
    if job_prediction:
        tabs.append("💼 Job Prediction")
    
    tab_objects = st.tabs(tabs)
    
    # Unpack tabs dynamically
    tab1 = tab_objects[0]
    tab2 = tab_objects[1]
    tab3 = tab_objects[2]
    tab4 = tab_objects[3]
    tab5 = tab_objects[4]
    tab6 = tab_objects[5] if len(tab_objects) > 5 else None
    
    with tab1:
        st.subheader("Skills Analysis")
        
        matched = ats_results['similarity_scores']['matched_skills']
        missing = ats_results['similarity_scores']['missing_skills']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Matched Skills")
            if matched:
                for skill in matched:
                    st.markdown(f"- {skill}")
            else:
                st.info("No matched skills found")
        
        with col2:
            st.markdown("### ❌ Missing Skills")
            if missing:
                for skill in missing:
                    st.markdown(f"- {skill}")
            else:
                st.success("No missing skills!")
    
    with tab2:
        st.subheader("Detailed Scoring Breakdown")
        
        scores = ats_results['similarity_scores']['detailed_scores']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Component Scores:**")
            st.progress(scores['skills_match_rate'], text=f"Skills Match: {scores['skills_match_rate']*100:.1f}%")
            st.progress(scores['all_keywords_match_rate'], text=f"Keyword Match: {scores['all_keywords_match_rate']*100:.1f}%")
            st.progress(scores['tfidf_match_rate'], text=f"TF-IDF Match: {scores['tfidf_match_rate']*100:.1f}%")
            st.progress(scores['text_similarity'], text=f"Text Similarity: {scores['text_similarity']*100:.1f}%")
        
        with col2:
            st.markdown("**Weights Used:**")
            st.code(f"""
Skills Match:     60% (when TF-IDF fails) / 40%
Keyword Match:    30% / 25%
TF-IDF Match:     0% (fallback) / 20%
Text Similarity:  10% / 15%

Note: Weights automatically adjust
if TF-IDF extraction fails.
            """)
            
            st.markdown("**Match Level:**")
            level = get_match_level(overall_match)
            if "Excellent" in level:
                st.success(f"✅ {level}")
            elif "Good" in level:
                st.info(f"ℹ️ {level}")
            elif "Moderate" in level:
                st.warning(f"⚠️ {level}")
            else:
                st.error(f"❌ {level}")
    
    with tab3:
        st.subheader("RAG Skills Extraction Results")
        
        if rag_skills:
            st.success(f"Detected {len(rag_skills)} skills using semantic similarity")
            
            # Group by confidence ranges
            very_high = [s for s in rag_skills[:50]]  # Top 50
            high = [s for s in rag_skills[50:150]]     # Next 100
            good = [s for s in rag_skills[150:]]       # Rest
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**🟢 Top Skills ({len(very_high)})**")
                for skill in very_high[:20]:
                    st.markdown(f"- {skill}")
                if len(very_high) > 20:
                    st.caption(f"...and {len(very_high)-20} more")
            
            with col2:
                st.markdown(f"**🟡 Strong Match ({len(high)})**")
                for skill in high[:20]:
                    st.markdown(f"- {skill}")
                if len(high) > 20:
                    st.caption(f"...and {len(high)-20} more")
            
            with col3:
                st.markdown(f"**🔵 Good Match ({len(good)})**")
                for skill in good[:20]:
                    st.markdown(f"- {skill}")
                if len(good) > 20:
                    st.caption(f"...and {len(good)-20} more")
        else:
            st.info("RAG analysis not enabled or no skills detected")
    
    with tab4:
        st.subheader("LLM Structured Analysis")
        
        if llm_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 💻 Technical Skills")
                tech_skills = llm_analysis.get('technical_skills', [])
                if tech_skills:
                    for skill in tech_skills:
                        st.markdown(f"- {skill}")
                else:
                    st.info("No technical skills extracted")
                
                st.markdown("### 📜 Certifications")
                certs = llm_analysis.get('certifications', [])
                if certs:
                    for cert in certs:
                        st.markdown(f"- {cert}")
                else:
                    st.info("No certifications found")
            
            with col2:
                st.markdown("### 🤝 Soft Skills")
                soft_skills = llm_analysis.get('soft_skills', [])
                if soft_skills:
                    for skill in soft_skills:
                        st.markdown(f"- {skill}")
                else:
                    st.info("No soft skills extracted")
                
                st.markdown("### 💼 Experience")
                st.metric("Total Years", f"{llm_analysis.get('total_experience_years', 0)} years")
                
                st.markdown("### 🎓 Education")
                education = llm_analysis.get('education', [])
                if education:
                    for edu in education:
                        degree = edu.get('degree', 'Unknown')
                        field = edu.get('field_of_study', '')
                        institution = edu.get('institution', '')
                        st.markdown(f"- **{degree}** {field}")
                        if institution:
                            st.caption(f"  {institution}")
                else:
                    st.info("No education information extracted")
        else:
            st.info("LLM analysis not enabled")
    
    with tab5:
        st.subheader("Resume Format Analysis")
        
        format_info = ats_results['format_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ATS Score", f"{format_info['ats_friendly_score']}/100")
            st.markdown(f"**Rating:** {format_info['ats_friendly_rating']}")
            
            st.markdown("### 📋 Sections Detected")
            sections = format_info.get('detected_sections', [])
            if sections:
                for section in sections:
                    st.markdown(f"✓ {section.title()}")
            else:
                st.warning("No clear sections detected")
        
        with col2:
            st.markdown("### 📞 Contact Information")
            contact = format_info['has_contact_info']
            st.markdown(f"{'✅' if contact['email'] else '❌'} Email")
            st.markdown(f"{'✅' if contact['phone'] else '❌'} Phone")
            st.markdown(f"{'✅' if contact['linkedin'] else '❌'} LinkedIn")
            st.markdown(f"{'✅' if contact['github'] else '❌'} GitHub")
            
            st.markdown("### 📊 Formatting")
            formatting = format_info['formatting']
            st.markdown(f"Bullet Points: {'✅' if formatting['has_bullets'] else '❌'}")
            st.markdown(f"Dates Found: {formatting['dates_found']}")
            st.markdown(f"Lines: {formatting['non_empty_lines']}")
        
        if format_info.get('quality_issues'):
            st.warning("**Quality Issues:**")
            for issue in format_info['quality_issues']:
                st.markdown(f"⚠️ {issue}")
    
    if tab6 and job_prediction:
        with tab6:
            st.subheader("AI Job Role Prediction")
            st.info("Based on your skills, our AI model predicts the following job roles:")
            
            # Main prediction
            st.markdown(f"### 🎯 Best Match: **{job_prediction['predicted_role']}**")
            st.progress(job_prediction['confidence'], text=f"Confidence: {job_prediction['confidence']:.1%}")
            
            # Top 5 predictions
            st.markdown("### 📊 Top 5 Predicted Roles:")
            for i, (role, prob) in enumerate(job_prediction['top_predictions'], 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{i}. {role}**")
                with col2:
                    st.markdown(f"{prob:.1%}")
                st.progress(prob, text="")
            
            # All probabilities chart
            st.markdown("### 📈 All Role Probabilities:")
            all_probs = job_prediction['all_probabilities']
            sorted_roles = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            
            for role, prob in sorted_roles:
                st.progress(prob, text=f"{role}: {prob:.1%}")

def get_match_level(score):
    """Get match level text"""
    if score >= 80:
        return "Excellent Match"
    elif score >= 65:
        return "Good Match"
    elif score >= 50:
        return "Moderate Match"
    else:
        return "Poor Match"

if __name__ == "__main__":
    main()
