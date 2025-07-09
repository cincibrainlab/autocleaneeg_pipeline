"""
Authentication utilities for FDA 21 CFR Part 11 compliance mode.

This module provides Auth0-based user authentication for AutoClean EEG processing
with tamper-proof audit trails and electronic signatures.
"""

import json
import os
import secrets
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
from cryptography.fernet import Fernet
from platformdirs import user_config_dir

# Try to import optional dependencies
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    from ulid import ULID
    ULID_AVAILABLE = True
except ImportError:
    ULID_AVAILABLE = False

# Import local modules
from autoclean.utils.audit import get_user_context
from autoclean.utils.config import is_compliance_mode_enabled
from autoclean.utils.database import manage_database_conditionally
from autoclean.utils.logging import message


class AuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback during authentication flow."""

    def do_GET(self) -> None:
        """Handle GET request from Auth0 callback."""
        try:
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)

            if parsed_url.path == "/callback":
                # Extract authorization code from callback
                if "code" in query_params:
                    self.server.auth_code = query_params["code"][0]
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
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
                elif "error" in query_params:
                    error = query_params["error"][0]
                    error_description = query_params.get(
                        "error_description", ["Unknown error"]
                    )[0]
                    self.server.auth_error = f"{error}: {error_description}"

                    self.send_response(400)
                    self.send_header("Content-type", "text/html")
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
            message("error", f"Error handling OAuth callback: {e}")
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

    def configure_auth0(
        self,
        domain: str,
        client_id: str,
        client_secret: str,
        audience: Optional[str] = None,
    ) -> None:
        """
        Configure Auth0 application settings.

        Args:
            domain: Auth0 domain (e.g., 'your-tenant.auth0.com')
            client_id: Auth0 application client ID
            client_secret: Auth0 application client secret
            audience: Auth0 API audience (optional)
        """
        self.domain = domain.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.audience = audience or f"https://{self.domain}/api/v2/"

        # Save configuration
        config_data = {
            "domain": self.domain,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "audience": self.audience,
            "configured_at": datetime.now().isoformat(),
        }

        with open(self.config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        message("debug", f"Auth0 configuration saved for domain: {self.domain}")

    def configure_developer_auth0(self) -> None:
        """
        Configure Auth0 using developer-managed credentials.

        This method sets up Auth0 with credentials managed by the
        AutoClean developers, providing a seamless experience for users.
        Credentials are loaded from environment variables or secure endpoint.
        """
        credentials = self._load_developer_credentials()

        if not credentials:
            raise ValueError("Failed to load developer Auth0 credentials")

        self.domain = credentials["domain"]
        self.client_id = credentials["client_id"]
        self.client_secret = credentials["client_secret"]
        self.audience = credentials.get("audience") or f"https://{self.domain}/api/v2/"

        # Save configuration
        config_data = {
            "domain": self.domain,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "audience": self.audience,
            "configured_at": datetime.now().isoformat(),
            "managed_by": "developer",
            "source": credentials.get("source", "unknown"),
        }

        with open(self.config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        message("debug", f"Developer Auth0 configuration set for domain: {self.domain}")

    def _load_developer_credentials(self) -> Optional[Dict[str, str]]:
        """
        Load developer Auth0 credentials from available sources.

        Priority order:
        1. Secure endpoint (future implementation)
        2. Environment variables
        3. Fallback defaults (development only)

        Returns:
            Dictionary with Auth0 credentials or None if unavailable
        """
        # Future: Try secure endpoint first
        credentials = self._fetch_credentials_from_endpoint()
        if credentials:
            return {**credentials, "source": "secure_endpoint"}

        # Try environment variables
        credentials = self._load_credentials_from_env()
        if credentials:
            return {**credentials, "source": "environment"}

        # Development fallback (should not be used in production)
        credentials = self._get_fallback_credentials()
        if credentials:
            message(
                "warning", "Using fallback Auth0 credentials - not for production use"
            )
            return {**credentials, "source": "fallback"}

        return None

    def _fetch_credentials_from_endpoint(self) -> Optional[Dict[str, str]]:
        """
        Fetch Auth0 credentials from secure endpoint.

        Future implementation will retrieve credentials from a secure
        endpoint with proper authentication and encryption.

        Returns:
            Dictionary with Auth0 credentials or None if unavailable
        """
        # TODO: Implement secure endpoint fetching
        # Example implementation:
        # try:
        #     endpoint_url = os.getenv("AUTOCLEAN_CREDENTIALS_ENDPOINT")
        #     if not endpoint_url:
        #         return None
        #
        #     response = requests.get(
        #         endpoint_url,
        #         headers={"Authorization": f"Bearer {self._get_service_token()}"},
        #         timeout=10
        #     )
        #     response.raise_for_status()
        #     return response.json()
        # except Exception as e:
        #     message("debug", f"Failed to fetch credentials from endpoint: {e}")
        #     return None

        return None

    def _load_credentials_from_env(self) -> Optional[Dict[str, str]]:
        """
        Load Auth0 credentials from environment variables and .env files.

        Automatically loads .env files if available, then checks environment variables.
        This makes .env files work seamlessly without requiring manual sourcing.

        Expected environment variables:
        - AUTOCLEAN_AUTH0_DOMAIN
        - AUTOCLEAN_AUTH0_CLIENT_ID
        - AUTOCLEAN_AUTH0_CLIENT_SECRET
        - AUTOCLEAN_AUTH0_AUDIENCE (optional)

        Returns:
            Dictionary with Auth0 credentials or None if unavailable
        """
        # Always try to load .env file first - this is the primary method
        self._load_dotenv_if_available()

        domain = os.getenv("AUTOCLEAN_AUTH0_DOMAIN")
        client_id = os.getenv("AUTOCLEAN_AUTH0_CLIENT_ID")
        client_secret = os.getenv("AUTOCLEAN_AUTH0_CLIENT_SECRET")
        audience = os.getenv("AUTOCLEAN_AUTH0_AUDIENCE")

        if not all([domain, client_id, client_secret]):
            message(
                "debug",
                "Missing required Auth0 environment variables (checked both environment and .env files)",
            )
            return None

        credentials = {
            "domain": domain,
            "client_id": client_id,
            "client_secret": client_secret,
        }

        if audience:
            credentials["audience"] = audience

        message(
            "debug",
            "Loaded Auth0 credentials from environment (including .env files if present)",
        )
        return credentials

    def _load_dotenv_if_available(self) -> None:
        """
        Automatically load .env files to make credentials seamlessly available.

        This is the primary method for loading credentials in development and
        production environments. Searches multiple common .env file locations.
        """
        if not DOTENV_AVAILABLE:
            return None

        try:
            # Look for .env file in current directory and parent directories
            env_paths = [
                Path(".env"),
                Path(".env.local"),
                Path("../.env"),
                Path("../../.env"),
                # Also check in the package directory
                Path(__file__).parent.parent.parent / ".env",
            ]

            loaded_files = []
            for env_path in env_paths:
                if env_path.exists():
                    try:
                        load_dotenv(
                            env_path, override=False
                        )  # Don't override existing env vars
                        loaded_files.append(str(env_path))
                        message("debug", f"Loaded .env file: {env_path}")
                    except Exception as e:
                        message("debug", f"Failed to load {env_path}: {e}")

            if loaded_files:
                message(
                    "debug", f"Successfully loaded {len(loaded_files)} .env file(s)"
                )
            else:
                message("debug", "No .env files found in search paths")

        except ImportError:
            # python-dotenv not available - this is expected in some environments
            message(
                "debug",
                "python-dotenv not installed - only system environment variables will be used",
            )
        except Exception as e:
            message("debug", f"Error during .env file loading: {e}")

    def _get_fallback_credentials(self) -> Optional[Dict[str, str]]:
        """
        Get fallback Auth0 credentials for development.

        These should only be used during development and testing.
        Production deployments should use environment variables or
        secure endpoint fetching.

        Returns:
            Dictionary with fallback credentials or None
        """
        # Only provide fallback in development environments
        if os.getenv("AUTOCLEAN_DEVELOPMENT_MODE") == "true":
            return {
                "domain": "autoclean-eeg-dev.us.auth0.com",
                "client_id": "dev_client_id_placeholder",
                "client_secret": "dev_client_secret_placeholder",
            }

        return None

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
            message(
                "error",
                "Auth0 not configured. Run 'autoclean-eeg setup --compliance-mode' first.",
            )
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

            message("info", "Opening browser for Auth0 authentication...")

            # Start local callback server
            server = HTTPServer(("localhost", 8080), AuthCallbackHandler)
            server.auth_code = None
            server.auth_error = None
            server.timeout = 1  # 1 second timeout for server operations

            # Open browser first
            webbrowser.open(auth_url)

            # Wait for callback (max 2 minutes)
            start_time = time.time()
            while time.time() - start_time < 120:
                try:
                    server.handle_request()
                    # Check if we got what we need
                    if server.auth_code or server.auth_error:
                        break
                except OSError:
                    # Timeout or other socket error, continue trying
                    pass
                time.sleep(0.1)

            try:
                server.server_close()
            except Exception:
                pass

            if server.auth_error:
                message("error", f"Authentication failed: {server.auth_error}")
                return False

            if not server.auth_code:
                message("error", "Authentication timed out. Please try again.")
                return False

            # Exchange authorization code for tokens
            return self._exchange_code_for_tokens(server.auth_code, redirect_uri)

        except Exception as e:
            message("error", f"Login failed: {e}")
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

        message("debug", "Logged out successfully")

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
            message("error", f"Failed to fetch user info: {e}")
            return None

    def refresh_access_token(self) -> bool:
        """
        Refresh access token using refresh token.

        Returns:
            True if refresh successful, False otherwise
        """
        if not self.refresh_token:
            message("warning", "No refresh token available")
            return False

        try:
            get_token = GetToken(
                self.domain, self.client_id, client_secret=self.client_secret
            )

            token_response = get_token.refresh_token(self.refresh_token)

            self.access_token = token_response["access_token"]
            expires_in = token_response.get("expires_in", 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

            # Update refresh token if provided
            if "refresh_token" in token_response:
                self.refresh_token = token_response["refresh_token"]

            self._save_tokens()
            message("debug", "Access token refreshed successfully")
            return True

        except Exception as e:
            message("error", f"Token refresh failed: {e}")
            return False

    def _exchange_code_for_tokens(self, auth_code: str, redirect_uri: str) -> bool:
        """Exchange authorization code for access and refresh tokens."""
        try:
            get_token = GetToken(
                self.domain, self.client_id, client_secret=self.client_secret
            )

            token_response = get_token.authorization_code(
                code=auth_code, redirect_uri=redirect_uri
            )

            self.access_token = token_response["access_token"]
            self.refresh_token = token_response.get("refresh_token")

            expires_in = token_response.get("expires_in", 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

            # Fetch user info
            self.current_user = None  # Clear cache to force refresh
            user_info = self.get_current_user()

            if user_info:
                message(
                    "debug",
                    f"Login successful for user: {user_info.get('email', 'Unknown')}",
                )
                self._save_tokens()
                return True
            else:
                message("error", "Failed to retrieve user information")
                return False

        except Auth0Error as e:
            message("error", f"Auth0 token exchange failed: {e}")
            return False
        except Exception as e:
            message("error", f"Token exchange failed: {e}")
            return False

    def _load_config(self) -> None:
        """Load Auth0 configuration from file."""
        if not self.config_file.exists():
            return

        try:
            with open(self.config_file, "r") as f:
                config_data = json.load(f)

            self.domain = config_data.get("domain")
            self.client_id = config_data.get("client_id")
            self.client_secret = config_data.get("client_secret")
            self.audience = config_data.get("audience")

        except Exception as e:
            message("error", f"Failed to load Auth0 config: {e}")

    def _save_tokens(self) -> None:
        """Save authentication tokens to encrypted file."""
        if not self.access_token:
            return

        token_data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": (
                self.token_expires_at.isoformat() if self.token_expires_at else None
            ),
            "user_info": self.current_user,
            "saved_at": datetime.now().isoformat(),
        }

        # Generate or load encryption key
        if not self.key_file.exists():
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)
            # Restrict file permissions
            os.chmod(self.key_file, 0o600)
        else:
            with open(self.key_file, "rb") as f:
                key = f.read()

        # Encrypt and save tokens
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(json.dumps(token_data).encode())

        with open(self.token_file, "wb") as f:
            f.write(encrypted_data)

        # Restrict file permissions
        os.chmod(self.token_file, 0o600)

    def _load_tokens(self) -> None:
        """Load authentication tokens from encrypted file."""
        if not self.token_file.exists() or not self.key_file.exists():
            return

        try:
            # Load encryption key
            with open(self.key_file, "rb") as f:
                key = f.read()

            # Load and decrypt tokens
            with open(self.token_file, "rb") as f:
                encrypted_data = f.read()

            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_data)
            token_data = json.loads(decrypted_data.decode())

            self.access_token = token_data.get("access_token")
            self.refresh_token = token_data.get("refresh_token")
            self.current_user = token_data.get("user_info")

            expires_at_str = token_data.get("expires_at")
            if expires_at_str:
                self.token_expires_at = datetime.fromisoformat(expires_at_str)

        except Exception as e:
            message("warning", f"Failed to load saved tokens: {e}")
            # Clean up corrupted token files
            for file_path in [self.token_file, self.key_file]:
                if file_path.exists():
                    file_path.unlink()


def get_auth0_manager() -> Auth0Manager:
    """Get singleton Auth0Manager instance."""
    if not hasattr(get_auth0_manager, "_instance"):
        get_auth0_manager._instance = Auth0Manager()
    return get_auth0_manager._instance




def validate_auth0_config(
    domain: str, client_id: str, client_secret: str
) -> Tuple[bool, str]:
    """
    Validate Auth0 configuration with basic format checking.

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

        # Domain format validation - be more flexible with Auth0 domains
        auth0_domains = [
            ".auth0.com",
            ".us.auth0.com",
            ".eu.auth0.com",
            ".au.auth0.com",
            ".jp.auth0.com",
        ]
        if not any(domain.endswith(suffix) for suffix in auth0_domains):
            return (
                False,
                "Auth0 domain must end with .auth0.com (or regional variant like .us.auth0.com)",
            )

        # Client ID format validation (Auth0 client IDs are typically alphanumeric)
        if (
            len(client_id) < 16
            or not client_id.replace("_", "").replace("-", "").isalnum()
        ):
            return False, "Auth0 Client ID format appears invalid"

        # Client secret length validation (Auth0 secrets are typically long)
        if len(client_secret) < 32:
            return False, "Auth0 Client Secret appears too short"

        # Try a simple connectivity test (optional - don't fail if network issues)
        try:
            response = requests.get(f"https://{domain}", timeout=5)
            if response.status_code in [
                200,
                404,
                403,
            ]:  # Any of these indicates the domain exists
                return True, "Auth0 configuration valid"
        except requests.RequestException:
            # Network issues - still proceed but warn
            pass

        # If we can't test connectivity, just validate format and proceed
        return True, "Auth0 configuration valid (network test skipped)"

    except Exception as e:
        return False, f"Configuration validation failed: {str(e)}"


def create_electronic_signature(
    run_id: str, signature_type: str = "processing_completion"
) -> Optional[str]:
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
        message("error", "Cannot create electronic signature: user not authenticated")
        return None

    user_info = auth_manager.get_current_user()
    if not user_info:
        message("error", "Cannot create electronic signature: user info unavailable")
        return None

    try:
        if not ULID_AVAILABLE:
            return None

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
            "auth_method": "auth0_oauth2",
        }

        # Store electronic signature
        signature_record = {
            "signature_id": signature_id,
            "run_id": run_id,
            "auth0_user_id": user_info.get("sub"),
            "signature_data": signature_data,
            "signature_type": signature_type,
        }

        manage_database_conditionally(
            "store_electronic_signature", signature_record
        )

        message(
            "debug", f"Electronic signature created: {signature_id} for run {run_id}"
        )
        return signature_id

    except Exception as e:
        message("error", f"Failed to create electronic signature: {e}")
        return None


def get_current_user_for_audit() -> Dict[str, Any]:
    """
    Get current user information for audit trail purposes.

    Returns:
        Dict with user information, or basic system info if not authenticated
    """
    if not is_compliance_mode_enabled():
        # Return basic system info for non-compliance mode
        return get_user_context()

    auth_manager = get_auth0_manager()

    if not auth_manager.is_authenticated():
        # Return basic info but mark as unauthenticated
        basic_context = get_user_context()
        basic_context["compliance_mode"] = True
        basic_context["authenticated"] = False
        return basic_context

    user_info = auth_manager.get_current_user()
    if not user_info:
        basic_context = get_user_context()
        basic_context["compliance_mode"] = True
        basic_context["authenticated"] = False
        return basic_context

    # Return enhanced user context for compliance mode
    basic_context = get_user_context()

    enhanced_context = {
        **basic_context,
        "compliance_mode": True,
        "authenticated": True,
        "auth0_user_id": user_info.get("sub"),
        "email": user_info.get("email"),
        "name": user_info.get("name"),
        "auth_provider": "auth0",
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
                message("error", "Compliance mode enabled but Auth0 not configured.")
                message(
                    "error",
                    "Run 'autoclean-eeg setup --compliance-mode' to configure authentication.",
                )
                return False

            if not auth_manager.is_authenticated():
                message("error", "Authentication required for compliance mode.")
                message("error", "Run 'autoclean-eeg login' to authenticate.")
                return False

            # Try to refresh token if needed
            if not auth_manager.is_authenticated() and auth_manager.refresh_token:
                if not auth_manager.refresh_access_token():
                    message("error", "Token refresh failed. Please login again.")
                    message("error", "Run 'autoclean-eeg login' to re-authenticate.")
                    return False

        return func(*args, **kwargs)

    return wrapper
