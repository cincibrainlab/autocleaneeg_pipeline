"""
Authentication utilities for FDA 21 CFR Part 11 compliance mode.

This module provides Auth0-based user authentication for AutoClean EEG processing
with tamper-proof audit trails and electronic signatures.
"""

import json
import os
import secrets
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests
from auth0.authentication import GetToken
from auth0.exceptions import Auth0Error
from auth0.management import Auth0
from cryptography.fernet import Fernet
from platformdirs import user_config_dir

from autoclean.utils.logging import configure_logger

logger = configure_logger(__name__)


class AuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback during authentication flow."""
    
    def do_GET(self) -> None:
        """Handle GET request from Auth0 callback."""
        try:
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)
            
            if parsed_url.path == '/callback':
                # Extract authorization code from callback
                if 'code' in query_params:
                    self.server.auth_code = query_params['code'][0]
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    success_html = """
                    <html>
                    <head><title>AutoClean Authentication</title></head>
                    <body>
                        <h2>✅ Authentication Successful!</h2>
                        <p>You can now close this browser window and return to the terminal.</p>
                        <script>setTimeout(function(){window.close();}, 3000);</script>
                    </body>
                    </html>
                    """
                    self.wfile.write(success_html.encode())
                elif 'error' in query_params:
                    error = query_params['error'][0]
                    error_description = query_params.get('error_description', ['Unknown error'])[0]
                    self.server.auth_error = f"{error}: {error_description}"
                    
                    self.send_response(400)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    error_html = f"""
                    <html>
                    <head><title>AutoClean Authentication Error</title></head>
                    <body>
                        <h2>❌ Authentication Failed</h2>
                        <p><strong>Error:</strong> {error}</p>
                        <p><strong>Description:</strong> {error_description}</p>
                        <p>Please close this window and try again.</p>
                    </body>
                    </html>  
                    """
                    self.wfile.write(error_html.encode())
                    
        except Exception as e:
            logger.error(f"Error handling OAuth callback: {e}")
            self.send_response(500)
            self.end_headers()
    
    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP server logging."""
        pass


class Auth0Manager:
    """
    Manages Auth0 authentication for FDA 21 CFR Part 11 compliance mode.
    
    This class handles:
    - OAuth 2.0 authorization code flow for CLI applications
    - Secure token storage with encryption
    - Automatic token refresh
    - User session management
    - Integration with AutoClean audit system
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize Auth0 manager.
        
        Args:
            config_dir: Directory for storing authentication config and tokens.
                       Defaults to user config directory.
        """
        self.config_dir = config_dir or Path(user_config_dir("autoclean"))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "auth_config.json"
        self.token_file = self.config_dir / "auth_tokens.enc"
        self.key_file = self.config_dir / "auth_key.key"
        
        # Auth0 configuration
        self.domain: Optional[str] = None
        self.client_id: Optional[str] = None
        self.client_secret: Optional[str] = None
        self.audience: Optional[str] = None
        
        # Current session
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.current_user: Optional[Dict[str, Any]] = None
        
        # Load existing configuration
        self._load_config()
        self._load_tokens()
    
    def configure_auth0(self, domain: str, client_id: str, client_secret: str, 
                       audience: Optional[str] = None) -> None:
        """
        Configure Auth0 application settings.
        
        Args:
            domain: Auth0 domain (e.g., 'your-tenant.auth0.com')
            client_id: Auth0 application client ID
            client_secret: Auth0 application client secret  
            audience: Auth0 API audience (optional)
        """
        self.domain = domain.rstrip('/')
        self.client_id = client_id
        self.client_secret = client_secret
        self.audience = audience or f"https://{self.domain}/api/v2/"
        
        # Save configuration
        config_data = {
            "domain": self.domain,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "audience": self.audience,
            "configured_at": datetime.now().isoformat()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Auth0 configuration saved for domain: {self.domain}")
    
    def is_configured(self) -> bool:
        """Check if Auth0 is properly configured."""
        return all([self.domain, self.client_id, self.client_secret])
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated with valid token."""
        if not self.access_token or not self.token_expires_at:
            return False
        
        # Check if token expires within next 5 minutes (buffer for API calls)
        buffer_time = datetime.now() + timedelta(minutes=5)
        return self.token_expires_at > buffer_time
    
    def login(self) -> bool:
        """
        Perform Auth0 login using OAuth 2.0 authorization code flow.
        
        Returns:
            True if login successful, False otherwise
        """
        if not self.is_configured():
            logger.error("Auth0 not configured. Run 'autoclean setup --compliance-mode' first.")
            return False
        
        try:
            # Generate OAuth parameters
            state = secrets.token_urlsafe(32)
            redirect_uri = "http://localhost:8080/callback"
            
            # Build authorization URL
            auth_url = (
                f"https://{self.domain}/authorize?"
                f"response_type=code&"
                f"client_id={self.client_id}&"
                f"redirect_uri={redirect_uri}&"
                f"scope=openid profile email&"
                f"audience={self.audience}&"
                f"state={state}"
            )
            
            logger.info("Opening browser for Auth0 authentication...")
            
            # Start local callback server
            server = HTTPServer(('localhost', 8080), AuthCallbackHandler)
            server.auth_code = None
            server.auth_error = None
            
            # Start server in background thread
            server_thread = threading.Thread(target=server.serve_request)
            server_thread.daemon = True
            server_thread.start()
            
            # Open browser
            webbrowser.open(auth_url)
            
            # Wait for callback (max 2 minutes)
            start_time = time.time()
            while time.time() - start_time < 120:
                if server.auth_code or server.auth_error:
                    break
                time.sleep(0.5)
            
            server.shutdown()
            server_thread.join(timeout=1)
            
            if server.auth_error:
                logger.error(f"Authentication failed: {server.auth_error}")
                return False
            
            if not server.auth_code:
                logger.error("Authentication timed out. Please try again.")
                return False
            
            # Exchange authorization code for tokens
            return self._exchange_code_for_tokens(server.auth_code, redirect_uri)
            
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def logout(self) -> None:
        """Clear authentication tokens and user session."""
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.current_user = None
        
        # Remove token files
        if self.token_file.exists():
            self.token_file.unlink()
        if self.key_file.exists():
            self.key_file.unlink()
        
        logger.info("Logged out successfully")
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """
        Get current authenticated user information.
        
        Returns:
            User information dict or None if not authenticated
        """
        if not self.is_authenticated():
            return None
        
        if self.current_user:
            return self.current_user
        
        try:
            # Fetch user info from Auth0
            userinfo_url = f"https://{self.domain}/userinfo"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = requests.get(userinfo_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            self.current_user = response.json()
            return self.current_user
            
        except Exception as e:
            logger.error(f"Failed to fetch user info: {e}")
            return None
    
    def refresh_access_token(self) -> bool:
        """
        Refresh access token using refresh token.
        
        Returns:
            True if refresh successful, False otherwise
        """
        if not self.refresh_token:
            logger.warning("No refresh token available")
            return False
        
        try:
            get_token = GetToken(self.domain, self.client_id, client_secret=self.client_secret)
            
            token_response = get_token.refresh_token(self.refresh_token)
            
            self.access_token = token_response['access_token']
            expires_in = token_response.get('expires_in', 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            # Update refresh token if provided
            if 'refresh_token' in token_response:
                self.refresh_token = token_response['refresh_token']
            
            self._save_tokens()
            logger.debug("Access token refreshed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False
    
    def _exchange_code_for_tokens(self, auth_code: str, redirect_uri: str) -> bool:
        """Exchange authorization code for access and refresh tokens."""
        try:
            get_token = GetToken(self.domain, self.client_id, client_secret=self.client_secret)
            
            token_response = get_token.authorization_code(
                code=auth_code,
                redirect_uri=redirect_uri
            )
            
            self.access_token = token_response['access_token']
            self.refresh_token = token_response.get('refresh_token')
            
            expires_in = token_response.get('expires_in', 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            # Fetch user info
            self.current_user = None  # Clear cache to force refresh
            user_info = self.get_current_user()
            
            if user_info:
                logger.info(f"Login successful for user: {user_info.get('email', 'Unknown')}")
                self._save_tokens()
                return True
            else:
                logger.error("Failed to retrieve user information")
                return False
                
        except Auth0Error as e:
            logger.error(f"Auth0 token exchange failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            return False
    
    def _load_config(self) -> None:
        """Load Auth0 configuration from file."""
        if not self.config_file.exists():
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            self.domain = config_data.get('domain')
            self.client_id = config_data.get('client_id')
            self.client_secret = config_data.get('client_secret')
            self.audience = config_data.get('audience')
            
        except Exception as e:
            logger.error(f"Failed to load Auth0 config: {e}")
    
    def _save_tokens(self) -> None:
        """Save authentication tokens to encrypted file."""
        if not self.access_token:
            return
        
        token_data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None,
            "user_info": self.current_user,
            "saved_at": datetime.now().isoformat()
        }
        
        # Generate or load encryption key
        if not self.key_file.exists():
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Restrict file permissions
            os.chmod(self.key_file, 0o600)
        else:
            with open(self.key_file, 'rb') as f:
                key = f.read()
        
        # Encrypt and save tokens
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(json.dumps(token_data).encode())
        
        with open(self.token_file, 'wb') as f:
            f.write(encrypted_data)
        
        # Restrict file permissions
        os.chmod(self.token_file, 0o600)
    
    def _load_tokens(self) -> None:
        """Load authentication tokens from encrypted file."""
        if not self.token_file.exists() or not self.key_file.exists():
            return
        
        try:
            # Load encryption key
            with open(self.key_file, 'rb') as f:
                key = f.read()
            
            # Load and decrypt tokens
            with open(self.token_file, 'rb') as f:
                encrypted_data = f.read()
            
            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_data)
            token_data = json.loads(decrypted_data.decode())
            
            self.access_token = token_data.get('access_token')
            self.refresh_token = token_data.get('refresh_token')
            self.current_user = token_data.get('user_info')
            
            expires_at_str = token_data.get('expires_at')
            if expires_at_str:
                self.token_expires_at = datetime.fromisoformat(expires_at_str)
            
        except Exception as e:
            logger.warning(f"Failed to load saved tokens: {e}")
            # Clean up corrupted token files
            for file_path in [self.token_file, self.key_file]:
                if file_path.exists():
                    file_path.unlink()


def get_auth0_manager() -> Auth0Manager:
    """Get singleton Auth0Manager instance."""
    if not hasattr(get_auth0_manager, '_instance'):
        get_auth0_manager._instance = Auth0Manager()
    return get_auth0_manager._instance


def is_compliance_mode_enabled() -> bool:
    """Check if compliance mode is enabled in user configuration."""
    try:
        from autoclean.utils.config import is_compliance_mode_enabled as config_compliance_check
        return config_compliance_check()
    except Exception:
        return False


def validate_auth0_config(domain: str, client_id: str, client_secret: str) -> Tuple[bool, str]:
    """
    Validate Auth0 configuration by testing connection.
    
    Args:
        domain: Auth0 domain
        client_id: Auth0 client ID  
        client_secret: Auth0 client secret
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Basic format validation
        if not domain or not client_id or not client_secret:
            return False, "All Auth0 configuration fields are required"
        
        # Domain format validation
        if not domain.endswith('.auth0.com') and not domain.endswith('.us.auth0.com') and not domain.endswith('.eu.auth0.com'):
            return False, "Auth0 domain must be in format 'your-tenant.auth0.com'"
        
        # Test connection to Auth0
        get_token = GetToken(domain, client_id, client_secret=client_secret)
        
        # Try to get a client credentials token
        try:
            audience = f"https://{domain}/api/v2/"
            token_response = get_token.client_credentials(audience)
            
            if 'access_token' in token_response:
                return True, "Auth0 configuration valid"
            else:
                return False, "Failed to obtain access token from Auth0"
                
        except Auth0Error as e:
            return False, f"Auth0 authentication failed: {str(e)}"
            
    except Exception as e:
        return False, f"Configuration validation failed: {str(e)}"


def create_electronic_signature(run_id: str, signature_type: str = "processing_completion") -> Optional[str]:
    """
    Create an electronic signature for a processing run.
    
    Args:
        run_id: The run ID to sign
        signature_type: Type of signature (e.g., 'processing_completion', 'data_review')
        
    Returns:
        Signature ID if successful, None otherwise
    """
    if not is_compliance_mode_enabled():
        return None
    
    auth_manager = get_auth0_manager()
    
    if not auth_manager.is_authenticated():
        logger.error("Cannot create electronic signature: user not authenticated")
        return None
    
    user_info = auth_manager.get_current_user()
    if not user_info:
        logger.error("Cannot create electronic signature: user info unavailable")
        return None
    
    try:
        from autoclean.utils.database import manage_database_with_audit_protection
        from python_ulid import ULID
        
        signature_id = str(ULID())
        current_time = datetime.now()
        
        # Create signature data
        signature_data = {
            "user_id": user_info.get("sub"),
            "email": user_info.get("email"),
            "name": user_info.get("name"),
            "timestamp": current_time.isoformat(),
            "signature_type": signature_type,
            "run_id": run_id,
            "user_agent": f"AutoClean-EEG/{signature_type}",
            "ip_address": "local",  # CLI application
            "auth_method": "auth0_oauth2"
        }
        
        # Store electronic signature
        signature_record = {
            "signature_id": signature_id,
            "run_id": run_id,
            "auth0_user_id": user_info.get("sub"),
            "signature_data": signature_data,
            "signature_type": signature_type
        }
        
        result = manage_database_with_audit_protection(
            "store_electronic_signature", 
            signature_record
        )
        
        logger.info(f"Electronic signature created: {signature_id} for run {run_id}")
        return signature_id
        
    except Exception as e:
        logger.error(f"Failed to create electronic signature: {e}")
        return None


def get_current_user_for_audit() -> Dict[str, Any]:
    """
    Get current user information for audit trail purposes.
    
    Returns:
        Dict with user information, or basic system info if not authenticated
    """
    if not is_compliance_mode_enabled():
        # Return basic system info for non-compliance mode
        from autoclean.utils.audit import get_user_context
        return get_user_context()
    
    auth_manager = get_auth0_manager()
    
    if not auth_manager.is_authenticated():
        # Return basic info but mark as unauthenticated
        from autoclean.utils.audit import get_user_context
        basic_context = get_user_context()
        basic_context["compliance_mode"] = True
        basic_context["authenticated"] = False
        return basic_context
    
    user_info = auth_manager.get_current_user()
    if not user_info:
        from autoclean.utils.audit import get_user_context
        basic_context = get_user_context()
        basic_context["compliance_mode"] = True
        basic_context["authenticated"] = False
        return basic_context
    
    # Return enhanced user context for compliance mode
    from autoclean.utils.audit import get_user_context
    basic_context = get_user_context()
    
    enhanced_context = {
        **basic_context,
        "compliance_mode": True,
        "authenticated": True,
        "auth0_user_id": user_info.get("sub"),
        "email": user_info.get("email"),
        "name": user_info.get("name"),
        "auth_provider": "auth0"
    }
    
    return enhanced_context


def require_authentication(func):
    """
    Decorator to require authentication for compliance mode operations.
    
    Usage:
        @require_authentication
        def protected_function():
            # This function requires authentication in compliance mode
            pass
    """
    def wrapper(*args, **kwargs):
        if is_compliance_mode_enabled():
            auth_manager = get_auth0_manager()
            
            if not auth_manager.is_configured():
                logger.error("Compliance mode enabled but Auth0 not configured.")
                logger.error("Run 'autoclean setup --compliance-mode' to configure authentication.")
                return False
            
            if not auth_manager.is_authenticated():
                logger.error("Authentication required for compliance mode.")
                logger.error("Run 'autoclean login' to authenticate.")
                return False
            
            # Try to refresh token if needed
            if not auth_manager.is_authenticated() and auth_manager.refresh_token:
                if not auth_manager.refresh_access_token():
                    logger.error("Token refresh failed. Please login again.")
                    logger.error("Run 'autoclean login' to re-authenticate.")
                    return False
        
        return func(*args, **kwargs)
    
    return wrapper