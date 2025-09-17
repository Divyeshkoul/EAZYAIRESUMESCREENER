import os
import base64
import json
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import streamlit as st

# Azure imports
from azure.storage.blob import BlobServiceClient

# Configure logging
logger = logging.getLogger(__name__)

class SimplifiedGmailService:
    """Simplified Gmail service for Streamlit Cloud compatibility"""
    
    def __init__(self, azure_connection_string: str, container_name: str = "resumes"):
        self.azure_connection_string = azure_connection_string
        self.container_name = container_name
        self.is_running = False
        self.status = {
            "last_sync": None,
            "emails_processed": 0,
            "files_uploaded": 0,
            "errors": [],
            "is_active": False,
            "auth_status": "Manual upload only"
        }
        
        # Show notice about Gmail integration
        self._show_gmail_notice()
    
    def _show_gmail_notice(self):
        """Show information about Gmail integration"""
        if not st.session_state.get("gmail_notice_shown", False):
            st.info("""
            📧 **Gmail Integration Notice**: 
            
            Gmail OAuth integration is available but may require manual setup on Streamlit Cloud.
            For now, you can use the file upload feature to process resumes directly.
            
            **Alternative options:**
            1. Use the file upload feature below
            2. For full Gmail integration, contact support
            """)
            st.session_state["gmail_notice_shown"] = True
    
    def connect_to_azure(self) -> Optional[BlobServiceClient]:
        """Connect to Azure Blob Storage"""
        try:
            blob_service_client = BlobServiceClient.from_connection_string(self.azure_connection_string)
            container_client = blob_service_client.get_container_client(self.container_name)
            container_client.get_container_properties()
            return blob_service_client
        except Exception as e:
            logger.error(f"Azure connection failed: {str(e)}")
            self.status["errors"].append(f"Azure connection failed: {str(e)}")
            return None
    
    def upload_manual_files(self, uploaded_files: List) -> Dict[str, any]:
        """Upload manually selected files to Azure"""
        if not uploaded_files:
            return {"error": "No files provided"}
        
        self.status["is_active"] = True
        self.status["errors"] = []
        uploaded_count = 0
        
        try:
            blob_service_client = self.connect_to_azure()
            if not blob_service_client:
                return self.get_status()
            
            container_client = blob_service_client.get_container_client(self.container_name)
            
            for uploaded_file in uploaded_files:
                try:
                    # Read file bytes
                    file_bytes = uploaded_file.read()
                    filename = uploaded_file.name
                    
                    # Check if supported format
                    supported_extensions = ['.pdf', '.docx', '.doc']
                    if any(filename.lower().endswith(ext) for ext in supported_extensions):
                        # Generate unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        unique_filename = f"{timestamp}_{filename}"
                        
                        # Upload to blob
                        blob_client = container_client.get_blob_client(unique_filename)
                        blob_client.upload_blob(file_bytes, overwrite=True)
                        
                        logger.info(f"Uploaded '{unique_filename}' to Azure")
                        uploaded_count += 1
                    else:
                        logger.warning(f"Skipping unsupported file: {filename}")
                        
                except Exception as e:
                    error_msg = f"Failed to upload {uploaded_file.name}: {str(e)}"
                    logger.error(error_msg)
                    self.status["errors"].append(error_msg)
            
            self.status["files_uploaded"] = uploaded_count
            self.status["last_sync"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
        except Exception as e:
            error_msg = f"Manual upload failed: {str(e)}"
            logger.error(error_msg)
            self.status["errors"].append(error_msg)
        finally:
            self.status["is_active"] = False
        
        return self.get_status()
    
    def get_status(self) -> Dict[str, any]:
        """Get current service status"""
        return self.status.copy()
    
    def render_upload_interface(self):
        """Render file upload interface"""
        st.markdown("### 📁 Manual File Upload")
        
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or DOC resume files"
        )
        
        if uploaded_files:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Selected files:** {len(uploaded_files)}")
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.size} bytes)")
            
            with col2:
                if st.button("📤 Upload to Azure", type="primary"):
                    with st.spinner("Uploading files..."):
                        result = self.upload_manual_files(uploaded_files)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.success(f"Uploaded {result.get('files_uploaded', 0)} files successfully!")
                            st.rerun()
        
        return uploaded_files

# Global service instance
simplified_gmail_service = None

def initialize_simplified_gmail_service(azure_connection_string: str) -> SimplifiedGmailService:
    """Initialize the simplified Gmail service"""
    global simplified_gmail_service
    simplified_gmail_service = SimplifiedGmailService(azure_connection_string)
    return simplified_gmail_service

def get_simplified_gmail_service() -> Optional[SimplifiedGmailService]:
    """Get the simplified Gmail service instance"""
    return simplified_gmail_service
