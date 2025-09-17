import streamlit as st
import pandas as pd
import base64
import asyncio
from datetime import datetime
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uuid
import time
import logging

# Configure logging for performance tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import your existing modules
from constants import AZURE_CONFIG
from utils import (
    parse_resume,
    get_text_chunks,
    get_embedding_cached,
    get_cosine_similarity,
    upload_to_blob,
    extract_contact_info,
    save_summary_to_blob,
    save_csv_to_blob
)
from backend import get_resume_analysis_async, extract_role_from_jd
from pdf_utils import generate_summary_pdf
from email_generator import send_email, check_missing_info, send_missing_info_email

# Import Simplified Gmail service
from simplified_gmail_service import initialize_simplified_gmail_service, get_simplified_gmail_service

# Azure Blob Storage
from azure.storage.blob import BlobServiceClient

# Enhanced Design with fixed light mode styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0f1419 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Light mode compatibility */
    [data-theme="light"] .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2a3a 0%, #0f1419 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    [data-theme="light"] section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.1);
    }

    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #00d4ff 0%, #5865f2 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.4rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    [data-theme="light"] .subtitle {
        color: #475569;
    }

    .candidate-card {
        background: linear-gradient(135deg, rgba(30, 42, 58, 0.8) 0%, rgba(15, 20, 25, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        color: #e2e8f0;
    }

    [data-theme="light"] .candidate-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
        border: 1px solid rgba(0, 0, 0, 0.1);
        color: #1e293b;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .candidate-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
        border-color: rgba(0, 212, 255, 0.3);
    }

    .metric-container {
        background: rgba(30, 42, 58, 0.6);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e2e8f0;
    }

    [data-theme="light"] .metric-container {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(0, 0, 0, 0.1);
        color: #1e293b;
    }

    .status-shortlist {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .status-review {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .status-reject {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .sidebar-section {
        background: rgba(0, 212, 255, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 3px solid #00d4ff;
        color: #e2e8f0;
    }

    [data-theme="light"] .sidebar-section {
        background: rgba(0, 212, 255, 0.1);
        color: #1e293b;
    }

    .performance-metrics {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #e2e8f0;
    }

    [data-theme="light"] .performance-metrics {
        color: #1e293b;
    }

    .upload-section {
        background: rgba(88, 101, 242, 0.1);
        border: 1px solid rgba(88, 101, 242, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #e2e8f0;
    }

    [data-theme="light"] .upload-section {
        color: #1e293b;
        background: rgba(88, 101, 242, 0.1);
    }

    .candidate-name {
        color: #e2e8f0;
        font-weight: 600;
    }

    [data-theme="light"] .candidate-name {
        color: #1e293b;
    }

    .candidate-contact {
        color: #94a3b8;
    }

    [data-theme="light"] .candidate-contact {
        color: #64748b;
    }

    .candidate-fitment {
        color: #cbd5e1;
    }

    [data-theme="light"] .candidate-fitment {
        color: #475569;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #5865f2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
def initialize_session_state():
    if "candidate_df" not in st.session_state:
        st.session_state["candidate_df"] = None
    if "analysis_done" not in st.session_state:
        st.session_state["analysis_done"] = False
    if "processing_metrics" not in st.session_state:
        st.session_state["processing_metrics"] = {}
    if "candidate_updates" not in st.session_state:
        st.session_state["candidate_updates"] = {}
    if "simplified_service_initialized" not in st.session_state:
        st.session_state["simplified_service_initialized"] = False

initialize_session_state()

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="EazyAI Resume Screener",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# Initialize BlobServiceClient
@st.cache_resource
def get_blob_service_client():
    return BlobServiceClient.from_connection_string(AZURE_CONFIG["connection_string"])

blob_service_client = get_blob_service_client()
resumes_container_client = blob_service_client.get_container_client(AZURE_CONFIG["resumes_container"])

# Initialize Simplified Gmail service
@st.cache_resource
def initialize_simplified_service():
    """Initialize simplified service for Streamlit Cloud"""
    try:
        service = initialize_simplified_gmail_service(AZURE_CONFIG["connection_string"])
        logger.info("Simplified Gmail service initialized")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize simplified service: {str(e)}")
        return None

# Start service
simplified_service = initialize_simplified_service()
if simplified_service and not st.session_state["simplified_service_initialized"]:
    st.session_state["simplified_service_initialized"] = True

def download_all_supported_resume_blobs():
    """Download all supported resume files (PDF, DOCX, DOC) from Azure Blob Storage"""
    try:
        blobs = resumes_container_client.list_blobs()
        resume_files = []
        supported_extensions = ['.pdf', '.docx', '.doc']
        
        for blob in blobs:
            if any(blob.name.lower().endswith(ext) for ext in supported_extensions):
                try:
                    downloader = resumes_container_client.download_blob(blob.name)
                    file_bytes = downloader.readall()
                    resume_files.append((blob.name, file_bytes))
                except Exception as e:
                    logger.error(f"Error downloading {blob.name}: {str(e)}")
                    continue
        
        logger.info(f"Downloaded {len(resume_files)} supported resume files")
        return resume_files
    except Exception as e:
        st.error(f"Error downloading from blob storage: {str(e)}")
        return []

def render_upload_status():
    """Render upload status and file management"""
    if simplified_service:
        status = simplified_service.get_status()
        
        st.markdown(f"""
        <div class="upload-section">
            <h4>📁 File Upload Status</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div>
                    <strong>Last Upload:</strong><br>
                    {status.get('last_sync', 'Never')}
                </div>
                <div>
                    <strong>Files Uploaded:</strong><br>
                    {status.get('files_uploaded', 0)}
                </div>
                <div>
                    <strong>Status:</strong><br>
                    {'🔄 Uploading...' if status.get('is_active') else '✅ Ready'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if status.get("errors"):
            with st.expander("⚠️ Upload Errors", expanded=False):
                for error in status["errors"][-3:]:
                    st.error(error)
        
        return status
    else:
        st.error("❌ Upload service unavailable")
        return {}

# Enhanced Header
st.markdown('<h1 class="main-title">EazyAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent Resume Screening Platform</p>', unsafe_allow_html=True)

# File Upload Section
col1, col2 = st.columns([2, 1])

with col1:
    upload_status = render_upload_status()

with col2:
    st.markdown("### 📤 Quick Upload")
    if simplified_service:
        # Render the upload interface
        uploaded_files = simplified_service.render_upload_interface()
    else:
        st.error("Upload service unavailable")

st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.markdown('<div class="sidebar-section"><h3>📋 Job Configuration</h3></div>', unsafe_allow_html=True)
    
    jd = st.text_area("📄 Paste Job Description", height=200, placeholder="Enter the complete job description here...")
    
    role = "N/A"
    if jd:
        with st.spinner("Extracting role from JD..."):
            role = extract_role_from_jd(jd)
            if role != "N/A":
                st.success(f"🎯 **Detected Role:** {role}")
            else:
                st.warning("⚠️ Could not extract role from JD")

    domain = st.text_input("🏢 Preferred Domain", placeholder="e.g., Healthcare, Fintech, E-commerce")
    skills = st.text_area("🛠️ Required Skills (comma separated)", placeholder="Python, React, AWS, Machine Learning")
    exp_range = st.selectbox("📈 Required Experience", ["0–1 yrs", "1–3 yrs", "2–4 yrs", "4+ yrs"])

    st.markdown('<div class="sidebar-section"><h3>🎚️ Matching Thresholds</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        jd_thresh = st.slider("JD Similarity", 0, 100, 60, help="Minimum similarity with job description")
        domain_thresh = st.slider("Domain Match", 0, 100, 50, help="Minimum domain experience match")
    with col2:
        skill_thresh = st.slider("Skills Match", 0, 100, 65, help="Minimum required skills match")
        exp_thresh = st.slider("Experience Match", 0, 100, 55, help="Experience level compatibility")
    
    shortlist_thresh = st.slider("🟢 Shortlist Threshold", 0, 100, 75, help="Score for automatic shortlisting")
    reject_thresh = st.slider("🔴 Reject Threshold", 0, 100, 40, help="Score below which candidates are rejected")
    top_n = st.number_input("🏆 Top-N Candidates", 0, 50, 0, help="Limit shortlisted candidates (0 = no limit)")

    st.markdown('<div class="sidebar-section"><h3>📂 Resume Source</h3></div>', unsafe_allow_html=True)
    
    load_from_blob = st.checkbox("☁️ Load from Azure Blob Storage", value=True, help="Load resumes from uploaded files")

    if not load_from_blob:
        uploaded_files = st.file_uploader(
            "📤 Upload Resume Files", 
            type=["pdf", "docx", "doc"], 
            accept_multiple_files=True,
            help="Select multiple resume files (PDF, DOCX, DOC formats supported)"
        )
    else:
        uploaded_files = None
        st.info("📊 Resumes will be loaded from Azure Blob Storage")

    st.markdown('<div class="sidebar-section"><h3>📁 File Management</h3></div>', unsafe_allow_html=True)
    
    if simplified_service:
        service_status = simplified_service.get_status()
        st.markdown(f"""
        **Files Uploaded:** {service_status.get('files_uploaded', 0)}  
        **Last Upload:** {service_status.get('last_sync', 'Never')}
        """)
        
        if service_status.get("errors"):
            st.warning(f"⚠️ {len(service_status['errors'])} upload errors")
    else:
        st.error("❌ File service unavailable")

    st.markdown("---")
    analyze = st.button("🚀 Start Analysis", type="primary", use_container_width=True)

# Main Processing Logic
if jd and analyze and not st.session_state["analysis_done"]:
    start_time = time.time()
    logger.info("Starting resume analysis")
    
    progress_container = st.container()
    with progress_container:
        st.markdown("### 🔄 Processing Resumes...")
        progress_bar = st.progress(0, text="Initializing analysis...")
        status_text = st.empty()

    # Load resumes
    if load_from_blob:
        status_text.info("📥 Loading resumes from Azure Blob Storage...")
        blob_files = download_all_supported_resume_blobs()
        total = len(blob_files)
        if total == 0:
            st.error("❌ No resume files found in Azure Blob storage container.")
            st.info("💡 **Tip:** Upload resume files using the upload section above!")
            st.stop()
        else:
            file_types = {}
            for file_name, _ in blob_files:
                ext = file_name.lower().split('.')[-1]
                file_types[ext] = file_types.get(ext, 0) + 1
            
            types_text = ", ".join([f"{count} {ext.upper()}" for ext, count in file_types.items()])
            st.info(f"📊 Found {total} resumes in blob storage ({types_text})")
    else:
        blob_files = None
        total = len(uploaded_files) if uploaded_files else 0
        if total == 0:
            st.error("❌ Please upload at least one resume or enable blob storage option.")
            st.stop()

    # Performance tracking
    processing_start = time.time()
    results = []
    
    # Pre-compute JD embedding once
    jd_embedding_start = time.time()
    jd_embedding = get_embedding_cached(jd)
    jd_embedding_time = time.time() - jd_embedding_start
    logger.info(f"JD embedding computed in {jd_embedding_time:.2f} seconds")

    async def process_all_resumes():
        tasks = []
        resume_processing_start = time.time()

        if load_from_blob:
            for idx, (file_name, file_bytes) in enumerate(blob_files):
                progress = (idx + 1) / total
                progress_bar.progress(progress, text=f"Processing {file_name} ({idx+1}/{total})")
                
                try:
                    upload_to_blob(file_bytes, file_name, AZURE_CONFIG["resumes_container"])
                    resume_text = parse_resume(file_bytes, file_name)
                    contact = extract_contact_info(resume_text)
                    chunks = get_text_chunks(resume_text)
                    resume_embedding = get_embedding_cached(" ".join(chunks[:3]))
                    jd_sim = round(get_cosine_similarity(resume_embedding, jd_embedding) * 100, 2)

                    task = get_resume_analysis_async(
                        jd=jd, resume_text=resume_text, contact=contact, role=role,
                        domain=domain, skills=skills, experience_range=exp_range,
                        jd_similarity=jd_sim, resume_file=file_name
                    )
                    tasks.append(task)
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {str(e)}")
                    continue
        else:
            for idx, file in enumerate(uploaded_files):
                progress = (idx + 1) / total
                progress_bar.progress(progress, text=f"Processing {file.name} ({idx+1}/{total})")
                
                try:
                    file_bytes = file.read()
                    file_name = file.name.replace(".pdf", "")
                    upload_to_blob(file_bytes, file_name + ".pdf", AZURE_CONFIG["resumes_container"])
                    resume_text = parse_resume(file_bytes, file.name)
                    contact = extract_contact_info(resume_text)
                    chunks = get_text_chunks(resume_text)
                    resume_embedding = get_embedding_cached(" ".join(chunks[:3]))
                    jd_sim = round(get_cosine_similarity(resume_embedding, jd_embedding) * 100, 2)

                    task = get_resume_analysis_async(
                        jd=jd, resume_text=resume_text, contact=contact, role=role,
                        domain=domain, skills=skills, experience_range=exp_range,
                        jd_similarity=jd_sim, resume_file=file_name
                    )
                    tasks.append(task)
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}")
                    continue

        if tasks:
            status_text.info("🧠 Running AI analysis on all resumes...")
            return await asyncio.gather(*tasks, return_exceptions=True)
        return []

    # Run async processing
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(process_all_resumes())
        loop.close()
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        logger.error(f"Processing error: {str(e)}")
        st.stop()

    # Filter out exceptions and process results
    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Task failed: {str(r)}")
            continue
        if isinstance(r, dict):
            r["recruiter_notes"] = ""
            valid_results.append(r)

    if not valid_results:
        st.error("❌ No resumes were successfully processed.")
        st.stop()

    results = valid_results
    processing_time = time.time() - processing_start
    total_time = time.time() - start_time

    # Enhanced verdict logic
    def determine_verdict(row):
        score = row["score"]
        if (
            row["jd_similarity"] < jd_thresh or
            row["skills_match"] < skill_thresh or
            row["domain_match"] < domain_thresh or
            row["experience_match"] < exp_thresh or
            score < reject_thresh
        ):
            return "reject"
        elif score >= shortlist_thresh:
            return "shortlist"
        else:
            return "review"

    # Create DataFrame and apply verdict logic
    df = pd.DataFrame(results).fillna("N/A")
    df.replace("n/a", "N/A", regex=True, inplace=True)
    df["verdict"] = df.apply(determine_verdict, axis=1)

    # Apply Top-N logic
    if top_n > 0:
        sorted_df = df.sort_values("score", ascending=False)
        top_candidates = sorted_df.head(top_n).copy()
        top_candidates["verdict"] = "shortlist"
        remaining = sorted_df.iloc[top_n:].copy()
        df = pd.concat([top_candidates, remaining], ignore_index=True)

    # Store results and metrics
    st.session_state["candidate_df"] = df
    st.session_state["analysis_done"] = True
    st.session_state["processing_metrics"] = {
        "total_time": total_time,
        "processing_time": processing_time,
        "jd_embedding_time": jd_embedding_time,
        "resumes_processed": len(df),
        "avg_time_per_resume": processing_time / len(df) if len(df) > 0 else 0
    }

    progress_bar.progress(1.0, text="✅ Analysis completed!")
    
    metrics = st.session_state["processing_metrics"]
    st.markdown(f"""
    <div class="performance-metrics">
        <h4>⚡ Performance Metrics</h4>
        <ul>
            <li><strong>Total Processing Time:</strong> {metrics['total_time']:.2f} seconds</li>
            <li><strong>Resumes Processed:</strong> {metrics['resumes_processed']}</li>
            <li><strong>Average Time per Resume:</strong> {metrics['avg_time_per_resume']:.2f} seconds</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.success(f"🎉 Successfully processed {len(results)} resumes in {total_time:.2f} seconds!")
    logger.info(f"Analysis completed: {len(results)} resumes in {total_time:.2f} seconds")

# Display Results (same as before but simplified)
if st.session_state["candidate_df"] is not None:
    df = st.session_state["candidate_df"]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    shortlisted_count = len(df[df["verdict"] == "shortlist"])
    review_count = len(df[df["verdict"] == "review"]) 
    rejected_count = len(df[df["verdict"] == "reject"])
    total_count = len(df)
    
    with col1:
        st.markdown(f'<div class="metric-container"><h3>✅ {shortlisted_count}</h3><p>Shortlisted</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-container"><h3>🟨 {review_count}</h3><p>Under Review</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-container"><h3>❌ {rejected_count}</h3><p>Rejected</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-container"><h3>📊 {total_count}</h3><p>Total Processed</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Enhanced tabs
    tabs = st.tabs([
        f"✅ Shortlisted ({shortlisted_count})", 
        f"🟨 Under Review ({review_count})", 
        f"❌ Rejected ({rejected_count})", 
        "📊 Analytics Dashboard"
    ])

    # Simplified candidate rendering (keeping the essential functionality)
    def render_candidate_card(row, verdict, idx):
        with st.container():
            st.markdown('<div class="candidate-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                status_class = f"status-{verdict}"
                candidate_name = row.get('name', 'Unknown')
                st.markdown(f"""
                <div class="candidate-name">
                    <h3>{candidate_name} <span class="{status_class}">{verdict.upper()}</span></h3>
                </div>
                """, unsafe_allow_html=True)
                
                email = row.get('email', 'N/A')
                phone = row.get('phone', 'N/A')
                st.markdown(f'<div class="candidate-contact">📧 <strong>{email}</strong> | 📞 <strong>{phone}</strong></div>', unsafe_allow_html=True)
                
                fitment = row.get('fitment', 'N/A')
                if pd.isna(fitment) or fitment == '' or fitment == 'None':
                    fitment = 'Analysis pending'
                if len(str(fitment)) > 200:
                    fitment = str(fitment)[:200] + "..."
                st.markdown(f'<div class="candidate-fitment">💡 <strong>Fitment:</strong> {fitment}</div>', unsafe_allow_html=True)
                
                # Score metrics
                col_jd, col_skills, col_domain, col_exp, col_final = st.columns(5)
                
                def safe_get_score(key, default=0):
                    value = row.get(key, default)
                    try:
                        return int(float(value)) if pd.notna(value) else default
                    except (ValueError, TypeError):
                        return default
                
                with col_jd:
                    st.metric("JD Match", f"{safe_get_score('jd_similarity')}%")
                with col_skills:
                    st.metric("Skills", f"{safe_get_score('skills_match')}%")
                with col_domain:
                    st.metric("Domain", f"{safe_get_score('domain_match')}%")
                with col_exp:
                    st.metric("Experience", f"{safe_get_score('experience_match')}%")
                with col_final:
                    st.metric("Final Score", f"{safe_get_score('score')}%")

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

    # Process each verdict tab
    for verdict, tab in zip(["shortlist", "review", "reject"], tabs[:3]):
        with tab:
            filtered = df[df["verdict"] == verdict].copy()
            
            if len(filtered) == 0:
                st.info(f"No candidates in {verdict} category")
                continue
            
            # Display candidates
            for idx, (i, row) in enumerate(filtered.iterrows()):
                render_candidate_card(row, verdict, idx)

            # Export functionality
            if len(filtered) > 0:
                st.markdown("### 📤 Export Data")
                export_df = filtered.drop(columns=["resume_text", "embedding"], errors="ignore")
                csv_name = f"{verdict}_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    "📊 Download CSV", 
                    csv_data,
                    file_name=csv_name,
                    mime="text/csv"
                )

    # Analytics Dashboard Tab (simplified)
    with tabs[3]:
        st.markdown("### 📊 Analytics Dashboard")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Verdict Distribution")
            verdict_counts = df["verdict"].value_counts()
            st.bar_chart(verdict_counts)
            
        with col2:
            st.markdown("#### 🎯 Score Statistics")
            score_stats = df['score'].describe()
            st.write(f"**Average Score:** {score_stats['mean']:.1f}%")
            st.write(f"**Median Score:** {score_stats['50%']:.1f}%")
            st.write(f"**Highest Score:** {score_stats['max']:.1f}%")
            st.write(f"**Lowest Score:** {score_stats['min']:.1f}%")

elif not st.session_state["analysis_done"]:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; background: rgba(30, 42, 58, 0.6); border-radius: 16px; margin: 2rem 0;">
        <h2>🚀 Welcome to EazyAI Resume Screener</h2>
        <p style="font-size: 1.2rem; color: #94a3b8; margin-bottom: 2rem;">
            Streamline your hiring process with AI-powered resume analysis
        </p>
        <div style="background: rgba(0, 212, 255, 0.1); padding: 1.5rem; border-radius: 12px; margin: 2rem 0; border: 1px solid rgba(0, 212, 255, 0.2);">
            <h3>📁 File Upload Integration</h3>
            <p style="margin-bottom: 1rem;">Upload your resume files using the interface above</p>
            <p style="color: #00d4ff; font-size: 0.9rem;">
                Supported formats: PDF, DOCX, DOC
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👈 **Get Started:** Upload resume files, add job description, configure settings, then click 'Start Analysis'!")
