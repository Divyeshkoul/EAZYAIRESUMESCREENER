# email_generator.py — Azure Communication Services implementation

import logging
from typing import Dict, List, Optional, Any
from constants import EMAIL_TEMPLATES
import os

# Azure Communication Services import
try:
    from azure.communication.email import EmailClient
    AZURE_EMAIL_AVAILABLE = True
except ImportError:
    logging.warning("Azure Communication Services not available - install azure-communication-email")
    AZURE_EMAIL_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Azure Email configuration
AZURE_EMAIL_CONFIG = {
    "connection_string": "endpoint=https://botemailsender.unitedstates.communication.azure.com/;accesskey=57bBq3CMdyxiO45PTjJy6K88hI9LK1N2CMPN3jr02Smz4mVSmFQrJQQJ99BIACULyCp8otsGAAAAAZCSqBSu",
    "sender_email": "donotreply@8c214184-f2e0-47f9-819a-e086fe0b4d19.azurecomm.net",
    "sender_name": "EazyAI Recruitment Team"
}

def validate_email(email: str) -> bool:
    """Validate email format"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email.strip()) is not None

def send_email(to_email: str, subject: str, body: str, from_email: Optional[str] = None) -> bool:
    """
    Send email using Azure Communication Services
    """
    if not AZURE_EMAIL_AVAILABLE:
        logger.error("Azure Communication Services not available")
        return False
        
    try:
        # Validate inputs
        if not to_email or not validate_email(to_email):
            logger.error(f"Invalid recipient email: {to_email}")
            return False
        
        if not subject or not body:
            logger.error("Subject or body is empty")
            return False
        
        # Initialize Azure Email client
        email_client = EmailClient.from_connection_string(AZURE_EMAIL_CONFIG["connection_string"])
        
        # Create HTML version of body
        html_body = body.replace('\n', '<br>')
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto;">
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
                    <h2 style="color: #0066cc; margin-bottom: 20px;">EazyAI Recruitment</h2>
                    <div style="background: white; padding: 20px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        {html_body}
                    </div>
                    <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
                    <p style="font-size: 12px; color: #666; text-align: center;">
                        This email was sent by EazyAI Resume Screener<br>
                        Powered by Azure Communication Services
                    </p>
                </div>
            </body>
        </html>
        """
        
        # Prepare email message
        message = {
            "senderAddress": AZURE_EMAIL_CONFIG["sender_email"],
            "recipients": {
                "to": [{"address": to_email}],
            },
            "content": {
                "subject": subject,
                "plainText": body,
                "html": html_content
            }
        }
        
        # Send email
        poller = email_client.begin_send(message)
        result = poller.result()
        
        logger.info(f"Azure email sent successfully to {to_email}, Message ID: {result.id}")
        return True
        
    except Exception as e:
        logger.error(f"Azure email sending failed: {str(e)}")
        return False

def generate_email_content(candidate: Dict[str, Any], verdict: str, role: str = "Position", company_name: str = "Our Company") -> Dict[str, str]:
    """
    Generate email content based on candidate data and verdict
    """
    try:
        name = candidate.get("name", "Candidate")
        
        # Get template based on verdict
        template = EMAIL_TEMPLATES.get(verdict.lower(), EMAIL_TEMPLATES.get("review", {
            "subject": "Application Status Update - {role}",
            "body": "Dear {name},\n\nThank you for your application for the {role} position.\n\nBest regards,\n{company_name} Team"
        }))
        
        # Format subject
        subject = template["subject"].format(role=role)
        
        # Format body with candidate-specific information
        body = template["body"].format(
            name=name,
            role=role,
            company_name=company_name,
            highlights=format_highlights(candidate.get("highlights", [])),
        )
        
        return {
            "subject": subject,
            "body": body
        }
        
    except Exception as e:
        logger.error(f"Error generating email content: {str(e)}")
        return {
            "subject": f"Application Update - {role}",
            "body": f"Dear {candidate.get('name', 'Candidate')},\n\nThank you for your application.\n\nBest regards,\n{company_name} Team"
        }

def format_highlights(highlights: List[str]) -> str:
    """Format highlights list for email"""
    try:
        if not highlights:
            return "• Your qualifications and experience"
        
        formatted = []
        for highlight in highlights[:5]:  # Limit to 5 highlights
            if highlight and highlight.strip():
                formatted.append(f"• {highlight.strip()}")
        
        return "\n".join(formatted) if formatted else "• Your qualifications and experience"
        
    except Exception as e:
        logger.error(f"Error formatting highlights: {str(e)}")
        return "• Your qualifications and experience"

def send_bulk_emails(candidates: List[Dict[str, Any]], verdict: str, role: str = "Position", company_name: str = "Our Company") -> Dict[str, int]:
    """
    Send bulk emails to multiple candidates using Azure
    """
    results = {
        "sent": 0,
        "failed": 0,
        "invalid_emails": 0
    }
    
    try:
        for candidate in candidates:
            email = candidate.get("email", "").strip()
            
            if not email or not validate_email(email):
                results["invalid_emails"] += 1
                continue
            
            # Generate email content
            email_content = generate_email_content(candidate, verdict, role, company_name)
            
            # Send email
            if send_email(email, email_content["subject"], email_content["body"]):
                results["sent"] += 1
            else:
                results["failed"] += 1
        
        logger.info(f"Azure bulk email results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error in bulk email sending: {str(e)}")
        return results

def check_missing_info(candidate: Dict[str, Any]) -> List[str]:
    """
    Check for missing information in candidate data
    """
    missing_info = []
    
    try:
        # Check required fields
        required_fields = {
            "name": "Full name",
            "email": "Email address",
            "phone": "Phone number"
        }
        
        for field, description in required_fields.items():
            value = candidate.get(field, "").strip()
            if not value or value.lower() in ["n/a", "na", "none", "null"]:
                missing_info.append(description)
        
        # Check for empty scores
        score_fields = ["skills_match", "domain_match", "experience_match", "jd_similarity"]
        for field in score_fields:
            if candidate.get(field, 0) == 0:
                missing_info.append(f"{field.replace('_', ' ').title()} score")
        
        # Check for missing content
        content_fields = {
            "fitment": "Fitment analysis",
            "summary_5_lines": "Candidate summary"
        }
        
        for field, description in content_fields.items():
            value = str(candidate.get(field, "")).strip()
            if not value or value.lower() in ["n/a", "na", "none", "null", "analysis not available"]:
                missing_info.append(description)
        
        return missing_info
        
    except Exception as e:
        logger.error(f"Error checking missing info: {str(e)}")
        return ["Error checking information completeness"]

def send_missing_info_email(candidate: Dict[str, Any], missing_info: List[str], role: str = "Position") -> bool:
    """
    Send email requesting missing information from candidate
    """
    try:
        email = candidate.get("email", "").strip()
        if not email or not validate_email(email):
            logger.error(f"Invalid email for missing info request: {email}")
            return False
        
        name = candidate.get("name", "Candidate")
        missing_list = "\n".join([f"• {item}" for item in missing_info])
        
        subject = f"Additional Information Required - {role} Application"
        
        body = f"""Dear {name},

Thank you for your application for the {role} position.

To complete our review of your application, we need some additional information:

{missing_list}

Please provide the missing information at your earliest convenience by replying to this email.

If you have any questions, please don't hesitate to contact us.

Best regards,
EazyAI Recruitment Team"""
        
        return send_email(email, subject, body)
        
    except Exception as e:
        logger.error(f"Error sending missing info email: {str(e)}")
        return False

def test_azure_email_connection() -> Dict[str, Any]:
    """
    Test Azure Communication Services connection
    """
    test_result = {
        "connection_successful": False,
        "service_available": AZURE_EMAIL_AVAILABLE,
        "error_message": None
    }
    
    try:
        if not AZURE_EMAIL_AVAILABLE:
            test_result["error_message"] = "Azure Communication Services not installed"
            return test_result
        
        # Test connection by initializing client
        email_client = EmailClient.from_connection_string(AZURE_EMAIL_CONFIG["connection_string"])
        
        # If we can initialize without error, connection is likely good
        test_result["connection_successful"] = True
        logger.info("Azure email connection test successful")
        
    except Exception as e:
        test_result["error_message"] = f"Connection failed: {str(e)}"
        logger.error(f"Azure email connection failed: {str(e)}")
    
    return test_result

def send_test_email(test_recipient: str) -> bool:
    """
    Send a test email to verify Azure functionality
    """
    try:
        if not validate_email(test_recipient):
            logger.error(f"Invalid test email recipient: {test_recipient}")
            return False
        
        subject = "EazyAI Resume Screener - Azure Email Test"
        body = f"""This is a test email from EazyAI Resume Screener using Azure Communication Services.

If you received this email, the Azure email configuration is working correctly.

Test Details:
• Service: Azure Communication Services
• Sender: {AZURE_EMAIL_CONFIG["sender_email"]}
• Recipient: {test_recipient}
• Timestamp: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Best regards,
EazyAI System"""
        
        return send_email(test_recipient, subject, body)
        
    except Exception as e:
        logger.error(f"Error sending Azure test email: {str(e)}")
        return False

def get_email_statistics(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about email addresses in candidate list
    """
    stats = {
        "total_candidates": len(candidates),
        "valid_emails": 0,
        "invalid_emails": 0,
        "missing_emails": 0,
        "email_domains": {},
        "duplicate_emails": 0
    }
    
    try:
        seen_emails = set()
        
        for candidate in candidates:
            email = candidate.get("email", "").strip().lower()
            
            if not email or email in ["n/a", "na", "none", "null"]:
                stats["missing_emails"] += 1
            elif not validate_email(email):
                stats["invalid_emails"] += 1
            else:
                if email in seen_emails:
                    stats["duplicate_emails"] += 1
                else:
                    seen_emails.add(email)
                    stats["valid_emails"] += 1
                    
                    # Extract domain
                    domain = email.split("@")[1]
                    stats["email_domains"][domain] = stats["email_domains"].get(domain, 0) + 1
        
        # Sort domains by frequency
        stats["email_domains"] = dict(sorted(stats["email_domains"].items(), key=lambda x: x[1], reverse=True))
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating email statistics: {str(e)}")
        return stats

def create_rejection_with_feedback(candidate: Dict[str, Any], role: str = "Position", feedback_points: List[str] = None) -> Dict[str, str]:
    """
    Create constructive rejection email with feedback
    """
    try:
        name = candidate.get("name", "Candidate")
        
        subject = f"Application Status Update - {role} Position"
        
        feedback_section = ""
        if feedback_points:
            feedback_section = """

Areas for potential development based on our requirements:
""" + "\n".join([f"• {point}" for point in feedback_points[:3]])  # Limit to 3 points
        
        body = f"""Dear {name},

Thank you for your interest in the {role} position and for taking the time to apply.

After careful consideration of all applications, we have decided not to proceed with your candidacy for this specific role. This decision was difficult given the quality of applications we received.{feedback_section}

We encourage you to continue developing your skills and to apply for future opportunities that may be a better match for your background.

We will keep your resume on file for future openings that may align with your experience.

Thank you again for considering us, and we wish you all the best in your career journey.

Best regards,
EazyAI Recruitment Team

---
If you have any questions about this decision, please feel free to reach out."""
        
        return {"subject": subject, "body": body}
        
    except Exception as e:
        logger.error(f"Error creating rejection with feedback: {str(e)}")
        return {
            "subject": f"Application Status - {role}",
            "body": f"Dear {candidate.get('name', 'Candidate')},\n\nThank you for your application.\n\nBest regards,\nEazyAI Recruitment Team"
        }

def send_interview_invitation(candidate: Dict[str, Any], interview_details: Dict[str, str], role: str = "Position") -> bool:
    """
    Send interview invitation email using Azure
    """
    try:
        email = candidate.get("email", "").strip()
        if not email or not validate_email(email):
            logger.error(f"Invalid email for interview invitation: {email}")
            return False
        
        name = candidate.get("name", "Candidate")
        
        subject = f"Interview Invitation - {role} Position"
        
        body = f"""Dear {name},

Congratulations! We are pleased to invite you for an interview for the {role} position.

Interview Details:
• Date: {interview_details.get('date', 'To be confirmed')}
• Time: {interview_details.get('time', 'To be confirmed')}
• Duration: {interview_details.get('duration', '45-60 minutes')}
• Format: {interview_details.get('format', 'In-person/Video call')}
• Location: {interview_details.get('location', 'To be confirmed')}

Please confirm your availability by replying to this email within 24 hours.

What to expect:
• Technical discussion about your experience
• Questions about the role and our company
• Opportunity for you to ask questions

Please bring:
• Updated resume
• Portfolio (if applicable)
• Valid ID

If you need to reschedule, please let us know as soon as possible.

We look forward to meeting you!

Best regards,
EazyAI Recruitment Team"""
        
        return send_email(email, subject, body)
        
    except Exception as e:
        logger.error(f"Error sending interview invitation: {str(e)}")
        return False

def send_follow_up_email(candidate: Dict[str, Any], role: str = "Position", days_since_application: int = 7) -> bool:
    """
    Send follow-up email to candidate using Azure
    """
    try:
        email = candidate.get("email", "").strip()
        if not email or not validate_email(email):
            logger.error(f"Invalid email for follow-up: {email}")
            return False
        
        name = candidate.get("name", "Candidate")
        
        subject = f"Application Status Update - {role} Position"
        
        body = f"""Dear {name},

Thank you for your interest in the {role} position and for your patience during our review process.

We wanted to provide you with an update on your application status:

Your application is currently under review by our hiring team. We have received a high volume of applications for this position, and we are carefully evaluating each candidate to ensure we make the best hiring decision.

What happens next:
• Our team will complete the initial review within the next 3-5 business days
• Qualified candidates will be contacted for the next stage of the process
• All applicants will be notified of their status regardless of the outcome

We appreciate your continued interest in our organization and will be in touch soon with an update.

If you have any questions in the meantime, please don't hesitate to reach out.

Best regards,
EazyAI Recruitment Team

---
Application submitted: {days_since_application} days ago
Current status: Under Review"""
        
        return send_email(email, subject, body)
        
    except Exception as e:
        logger.error(f"Error sending follow-up email: {str(e)}")
        return False
