import os
import base64
import json
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import streamlit as st

# Gmail API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Azure imports
from azure.storage.blob import BlobServiceClient

# Configure logging
logger = logging.getLogger(__name__)

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify']

# Gmail credentials (from your secrets)
GMAIL_CREDENTIALS = {
    "installed": {
        "client_id": "784425861832-ls95dljjtk84olo9v71j7a1uto11n5nm.apps.googleusercontent.com",
        "project_id": "gmail-to-azure-472409", 
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": "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",
        "redirect_uris": ["http://localhost"]
    }
}

class OAuthGmailToAzureService:
    def __init__(self, azure_connection_string: str, container_name: str = "resumes"):
        self.azure_connection_string = azure_connection_string
        self.container_name = container_name
        self.is_running = False
        self.gmail_service = None
        self.status = {
            "last_sync": None,
            "emails_processed": 0,
            "files_uploaded": 0,
            "errors": [],
            "is_active": False,
            "auth_status": "Not authenticated"
        }
        
    def authenticate_gmail(self, force_reauth: bool = False) -> bool:
        """Authenticate with Gmail using OAuth 2.0"""
        try:
            creds = None
            
            # Load existing token if available
            if 'gmail_token' in st.session_state and not force_reauth:
                try:
                    token_data = st.session_state['gmail_token']
                    creds = Credentials.from_authorized_user_info(token_data, SCOPES)
                except Exception as e:
                    logger.warning(f"Failed to load existing token: {str(e)}")
                    creds = None
            
            # If no valid credentials, start OAuth flow
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        logger.info("Gmail token refreshed successfully")
                    except Exception as e:
                        logger.warning(f"Token refresh failed: {str(e)}")
                        creds = None
                
                if not creds:
                    # Create credentials.json temporarily
                    credentials_path = "temp_credentials.json"
                    with open(credentials_path, 'w') as f:
                        json.dump(GMAIL_CREDENTIALS, f)
                    
                    try:
                        # Start OAuth flow
                        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                        
                        # For Streamlit Cloud, we need to handle this differently
                        if self._is_streamlit_cloud():
                            return self._handle_streamlit_oauth(flow)
                        else:
                            # Local development
                            creds = flow.run_local_server(port=0)
                            
                    finally:
                        # Clean up temporary file
                        if os.path.exists(credentials_path):
                            os.remove(credentials_path)
            
            if creds and creds.valid:
                # Store token in session state
                st.session_state['gmail_token'] = {
                    'token': creds.token,
                    'refresh_token': creds.refresh_token,
                    'token_uri': creds.token_uri,
                    'client_id': creds.client_id,
                    'client_secret': creds.client_secret,
                    'scopes': creds.scopes
                }
                
                # Build Gmail service
                self.gmail_service = build('gmail', 'v1', credentials=creds)
                self.status["auth_status"] = "Authenticated"
                logger.info("Gmail authentication successful")
                return True
            else:
                self.status["auth_status"] = "Authentication failed"
                self.status["errors"].append("Gmail OAuth authentication failed")
                return False
                
        except Exception as e:
            error_msg = f"Gmail authentication error: {str(e)}"
            logger.error(error_msg)
            self.status["errors"].append(error_msg)
            self.status["auth_status"] = "Authentication error"
            return False
    
    def _is_streamlit_cloud(self) -> bool:
        """Check if running on Streamlit Cloud"""
        return os.environ.get('STREAMLIT_SHARING_MODE') == 'true' or 'streamlit.io' in os.environ.get('HOSTNAME', '')
    
    def _handle_streamlit_oauth(self, flow) -> bool:
        """Handle OAuth flow for Streamlit Cloud deployment"""
        try:
            # Generate authorization URL
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            # Display instructions to user
            st.error("🔐 **Gmail Authentication Required**")
            st.markdown(f"""
            **Step 1:** Click the link below to authorize Gmail access:
            
            [🔗 Authorize Gmail Access]({auth_url})
            
            **Step 2:** After authorization, copy the authorization code and paste it below:
            """)
            
            # Input field for authorization code
            auth_code = st.text_input("📝 Paste Authorization Code:", type="password")
            
            if auth_code:
                try:
                    # Exchange code for token
                    flow.fetch_token(code=auth_code)
                    creds = flow.credentials
                    
                    # Store credentials
                    st.session_state['gmail_token'] = {
                        'token': creds.token,
                        'refresh_token': creds.refresh_token,
                        'token_uri': creds.token_uri,
                        'client_id': creds.client_id,
                        'client_secret': creds.client_secret,
                        'scopes': creds.scopes
                    }
                    
                    # Build Gmail service
                    self.gmail_service = build('gmail', 'v1', credentials=creds)
                    self.status["auth_status"] = "Authenticated"
                    st.success("✅ Gmail authentication successful!")
                    st.rerun()
                    return True
                    
                except Exception as e:
                    st.error(f"❌ Authentication failed: {str(e)}")
                    return False
            
            return False
            
        except Exception as e:
            st.error(f"❌ OAuth setup failed: {str(e)}")
            return False
    
    def connect_to_azure(self) -> Optional[BlobServiceClient]:
        """Connect to Azure Blob Storage"""
        try:
            blob_service_client = BlobServiceClient.from_connection_string(self.azure_connection_string)
            # Test connection by getting container properties
            container_client = blob_service_client.get_container_client(self.container_name)
            container_client.get_container_properties()
            return blob_service_client
        except Exception as e:
            logger.error(f"Azure connection failed: {str(e)}")
            self.status["errors"].append(f"Azure connection failed: {str(e)}")
            return None
    
    def build_time_query(self, time_filter: Dict[str, any]) -> str:
        """Build Gmail search query with time filters"""
        query_parts = ["has:attachment"]
        
        # Add time-based filters
        if time_filter.get("filter_type") == "last_hours":
            hours = time_filter.get("hours", 24)
            # Gmail uses relative time queries
            if hours <= 24:
                query_parts.append("newer_than:1d")
            elif hours <= 168:  # 7 days
                query_parts.append("newer_than:7d")
            else:
                query_parts.append("newer_than:30d")
        
        elif time_filter.get("filter_type") == "date_range":
            start_date = time_filter.get("start_date")
            end_date = time_filter.get("end_date")
            
            if start_date:
                query_parts.append(f"after:{start_date.strftime('%Y/%m/%d')}")
            if end_date:
                query_parts.append(f"before:{end_date.strftime('%Y/%m/%d')}")
        
        elif time_filter.get("filter_type") == "unread_only":
            query_parts.append("is:unread")
        
        # Add email filter
        target_email = time_filter.get("target_email", "eazyai111@gmail.com")
        if target_email:
            query_parts.append(f"to:{target_email}")
        
        return " ".join(query_parts)
    
    def get_gmail_messages(self, time_filter: Dict[str, any]) -> List[Dict]:
        """Get Gmail messages based on time filter"""
        try:
            if not self.gmail_service:
                if not self.authenticate_gmail():
                    return []
            
            # Build search query
            query = self.build_time_query(time_filter)
            logger.info(f"Gmail search query: {query}")
            
            # Search for messages
            results = self.gmail_service.users().messages().list(
                userId='me', 
                q=query,
                maxResults=time_filter.get("max_emails", 50)
            ).execute()
            
            messages = results.get('messages', [])
            logger.info(f"Found {len(messages)} matching emails")
            
            return messages
            
        except HttpError as e:
            error_msg = f"Gmail API error: {str(e)}"
            logger.error(error_msg)
            self.status["errors"].append(error_msg)
            return []
        except Exception as e:
            error_msg = f"Error fetching Gmail messages: {str(e)}"
            logger.error(error_msg)
            self.status["errors"].append(error_msg)
            return []
    
    def process_message_attachments(self, message_id: str, blob_service_client: BlobServiceClient) -> int:
        """Process attachments from a specific Gmail message"""
        uploaded_count = 0
        
        try:
            # Get message details
            message = self.gmail_service.users().messages().get(
                userId='me', 
                id=message_id,
                format='full'
            ).execute()
            
            # Extract message info
            headers = message['payload'].get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            
            logger.info(f"Processing message: {subject} from {sender}")
            
            # Process parts recursively
            uploaded_count += self._process_parts(
                message['payload'], 
                message_id, 
                blob_service_client,
                subject
            )
            
            # Mark as read if configured
            if uploaded_count > 0:
                self.gmail_service.users().messages().modify(
                    userId='me',
                    id=message_id,
                    body={'removeLabelIds': ['UNREAD']}
                ).execute()
            
        except Exception as e:
            error_msg = f"Error processing message {message_id}: {str(e)}"
            logger.error(error_msg)
            self.status["errors"].append(error_msg)
        
        return uploaded_count
    
    def _process_parts(self, part: Dict, message_id: str, blob_service_client: BlobServiceClient, subject: str) -> int:
        """Recursively process message parts for attachments"""
        uploaded_count = 0
        
        # Check if this part has attachments
        if 'parts' in part:
            for subpart in part['parts']:
                uploaded_count += self._process_parts(subpart, message_id, blob_service_client, subject)
        
        # Check if this part is an attachment
        if part.get('filename') and part.get('body', {}).get('attachmentId'):
            filename = part['filename']
            attachment_id = part['body']['attachmentId']
            
            # Check if it's a supported format
            supported_extensions = ['.pdf', '.docx', '.doc']
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                try:
                    # Download attachment
                    attachment = self.gmail_service.users().messages().attachments().get(
                        userId='me',
                        messageId=message_id,
                        id=attachment_id
                    ).execute()
                    
                    # Decode attachment data
                    file_data = base64.urlsafe_b64decode(attachment['data'])
                    
                    # Generate unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_filename = f"{timestamp}_{filename}"
                    
                    # Upload to Azure Blob
                    container_client = blob_service_client.get_container_client(self.container_name)
                    blob_client = container_client.get_blob_client(unique_filename)
                    
                    blob_client.upload_blob(file_data, overwrite=True)
                    
                    logger.info(f"Uploaded '{unique_filename}' from message: {subject}")
                    uploaded_count += 1
                    
                except Exception as e:
                    error_msg = f"Failed to process attachment {filename}: {str(e)}"
                    logger.error(error_msg)
                    self.status["errors"].append(error_msg)
            else:
                logger.info(f"Skipping unsupported file: {filename}")
        
        return uploaded_count
    
    def process_emails_with_filter(self, time_filter: Dict[str, any]) -> Dict[str, any]:
        """Process emails based on time filter"""
        self.status["is_active"] = True
        self.status["errors"] = []
        processed_count = 0
        uploaded_count = 0
        
        try:
            # Authenticate if needed
            if not self.gmail_service:
                if not self.authenticate_gmail():
                    return self.get_status()
            
            # Connect to Azure
            blob_service_client = self.connect_to_azure()
            if not blob_service_client:
                return self.get_status()
            
            # Get messages based on filter
            messages = self.get_gmail_messages(time_filter)
            
            if not messages:
                self.status["last_sync"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.status["is_active"] = False
                return self.get_status()
            
            # Process each message
            for message in messages:
                try:
                    message_id = message['id']
                    files_uploaded = self.process_message_attachments(message_id, blob_service_client)
                    
                    processed_count += 1
                    uploaded_count += files_uploaded
                    
                except Exception as e:
                    error_msg = f"Error processing message: {str(e)}"
                    logger.error(error_msg)
                    self.status["errors"].append(error_msg)
                    continue
            
            # Update status
            self.status["emails_processed"] = processed_count
            self.status["files_uploaded"] = uploaded_count
            self.status["last_sync"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Gmail sync completed: {processed_count} emails, {uploaded_count} files uploaded")
            
        except Exception as e:
            error_msg = f"Gmail sync failed: {str(e)}"
            logger.error(error_msg)
            self.status["errors"].append(error_msg)
        
        finally:
            self.status["is_active"] = False
        
        return self.get_status()
    
    def get_status(self) -> Dict[str, any]:
        """Get current sync status"""
        return self.status.copy()
    
    def start_background_sync(self, time_filter: Dict[str, any]) -> None:
        """Start background Gmail sync with time filter"""
        if self.is_running:
            return
        
        def background_task():
            self.is_running = True
            try:
                self.process_emails_with_filter(time_filter)
            except Exception as e:
                logger.error(f"Background sync error: {str(e)}")
            finally:
                self.is_running = False
        
        thread = threading.Thread(target=background_task, daemon=True)
        thread.start()
    
    def sync_now(self, time_filter: Dict[str, any]) -> Dict[str, any]:
        """Manually trigger sync with time filter"""
        if self.is_running:
            return {"error": "Sync already in progress"}
        
        return self.process_emails_with_filter(time_filter)
    
    def reset_authentication(self):
        """Reset Gmail authentication"""
        if 'gmail_token' in st.session_state:
            del st.session_state['gmail_token']
        self.gmail_service = None
        self.status["auth_status"] = "Not authenticated"

# Global service instance
oauth_gmail_service = None

def initialize_oauth_gmail_service(azure_connection_string: str) -> OAuthGmailToAzureService:
    """Initialize the OAuth Gmail service"""
    global oauth_gmail_service
    oauth_gmail_service = OAuthGmailToAzureService(azure_connection_string)
    return oauth_gmail_service

def get_oauth_gmail_service() -> Optional[OAuthGmailToAzureService]:
    """Get the OAuth Gmail service instance"""
    return oauth_gmail_service
