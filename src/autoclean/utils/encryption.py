"""
Encryption utilities for AutoClean EEG output storage in compliance mode.

This module provides encryption and decryption services for storing sensitive
outputs in the database when Part 11 compliance mode is enabled. In non-compliance
mode, all operations pass through without encryption for normal filesystem storage.
"""

import hashlib
import json
import os
import zlib
from pathlib import Path
from typing import Any, Dict, Optional, Union

from autoclean.utils.config import is_compliance_mode_enabled
from autoclean.utils.logging import message


class EncryptionManager:
    """
    Manages encryption/decryption of AutoClean outputs in compliance mode.
    
    In compliance mode:
    - All outputs are encrypted before database storage
    - Decryption requires authentication
    - Keys are stored securely
    
    In normal mode:
    - All operations are pass-through (no encryption)
    - Outputs are stored normally on filesystem
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize encryption manager.
        
        Args:
            config_dir: Directory for storing encryption keys. 
                       Uses same directory as Auth0 config.
        """
        from platformdirs import user_config_dir
        
        self.config_dir = config_dir or Path(user_config_dir("autoclean"))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Encryption key files
        self.output_key_file = self.config_dir / "output_encryption.key"
        
        # Runtime encryption state
        self._fernet = None
        
    def is_encryption_enabled(self) -> bool:
        """Check if encryption should be used (compliance mode + authenticated)."""
        if not is_compliance_mode_enabled():
            return False
            
        # In compliance mode, check if user is authenticated
        try:
            from autoclean.utils.auth import get_auth0_manager
            auth_manager = get_auth0_manager()
            
            if not auth_manager.is_authenticated():
                message("warning", "Compliance mode enabled but user not authenticated - encryption disabled")
                return False
                
            return True
            
        except ImportError:
            # Auth0 not available - this is OK for testing/development
            message("debug", "Auth0 not available - encryption disabled")
            return False
        except Exception as e:
            message("error", f"Failed to check authentication status: {e}")
            return False
    
    def _get_or_create_key(self) -> bytes:
        """Get existing encryption key or create new one."""
        if self.output_key_file.exists():
            with open(self.output_key_file, "rb") as f:
                key = f.read()
        else:
            # Create new encryption key
            try:
                from cryptography.fernet import Fernet
                key = Fernet.generate_key()
            except ImportError:
                # Fallback key generation if cryptography not available
                import secrets
                key = secrets.token_bytes(32)
                
            with open(self.output_key_file, "wb") as f:
                f.write(key)
            # Restrict file permissions to owner only
            os.chmod(self.output_key_file, 0o600)
            message("debug", "Created new output encryption key")
            
        return key
    
    def _get_fernet(self):
        """Get Fernet instance for encryption/decryption."""
        if self._fernet is None:
            try:
                from cryptography.fernet import Fernet
                key = self._get_or_create_key()
                self._fernet = Fernet(key)
            except ImportError:
                raise ImportError("cryptography package required for encryption. Install with: pip install cryptography")
        return self._fernet
    
    def encrypt_output(self, data: Union[str, bytes, Dict[str, Any]], compress: bool = True) -> Optional[bytes]:
        """
        Encrypt output data for database storage.
        
        Args:
            data: Data to encrypt (string, bytes, or dict)
            compress: Whether to compress data before encryption
            
        Returns:
            Encrypted bytes if compliance mode enabled, None if normal mode
        """
        if not self.is_encryption_enabled():
            return None
            
        try:
            # Convert data to bytes
            if isinstance(data, dict):
                data_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                # Try to serialize other types
                data_bytes = json.dumps(data, default=str).encode('utf-8')
            
            # Optional compression
            if compress and len(data_bytes) > 1024:  # Only compress larger data
                compressed_data = zlib.compress(data_bytes)
                # Only use compression if it actually reduces size
                if len(compressed_data) < len(data_bytes):
                    data_bytes = compressed_data
                    compress = True
                else:
                    compress = False
            else:
                compress = False
            
            # Encrypt the data
            fernet = self._get_fernet()
            encrypted_data = fernet.encrypt(data_bytes)
            
            # Prepend compression flag
            flag = b'\x01' if compress else b'\x00'
            final_data = flag + encrypted_data
            
            message("debug", f"Encrypted {len(data_bytes)} bytes to {len(final_data)} encrypted bytes (compressed: {compress})")
            return final_data
            
        except Exception as e:
            message("error", f"Failed to encrypt output data: {e}")
            raise
    
    def decrypt_output(self, encrypted_data: bytes, output_type: str = "json") -> Any:
        """
        Decrypt output data from database storage.
        
        Args:
            encrypted_data: Encrypted bytes from database
            output_type: Expected output type ("json", "text", "bytes")
            
        Returns:
            Decrypted data in requested format
        """
        if not self.is_encryption_enabled():
            raise ValueError("Cannot decrypt data - compliance mode not enabled or user not authenticated")
            
        try:
            # Extract compression flag and encrypted data
            if len(encrypted_data) < 1:
                raise ValueError("Invalid encrypted data: too short")
                
            compress_flag = encrypted_data[0:1]
            actual_encrypted_data = encrypted_data[1:]
            is_compressed = compress_flag == b'\x01'
            
            # Decrypt the data
            fernet = self._get_fernet()
            decrypted_bytes = fernet.decrypt(actual_encrypted_data)
            
            # Decompress if needed
            if is_compressed:
                decrypted_bytes = zlib.decompress(decrypted_bytes)
            
            # Convert to requested format
            if output_type == "bytes":
                return decrypted_bytes
            elif output_type == "text":
                return decrypted_bytes.decode('utf-8')
            elif output_type == "json":
                return json.loads(decrypted_bytes.decode('utf-8'))
            else:
                # Default to text
                return decrypted_bytes.decode('utf-8')
                
        except Exception as e:
            message("error", f"Failed to decrypt output data: {e}")
            raise
    
    def encrypt_file(self, file_path: Path, compress: bool = True) -> Optional[bytes]:
        """
        Encrypt a file for database storage.
        
        Args:
            file_path: Path to file to encrypt
            compress: Whether to compress file before encryption
            
        Returns:
            Encrypted file content as bytes, or None if not in compliance mode
        """
        if not self.is_encryption_enabled():
            return None
            
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            with open(file_path, "rb") as f:
                file_data = f.read()
            
            return self.encrypt_output(file_data, compress=compress)
            
        except Exception as e:
            message("error", f"Failed to encrypt file {file_path}: {e}")
            raise
    
    def decrypt_to_file(self, encrypted_data: bytes, output_path: Path) -> None:
        """
        Decrypt data and save to file.
        
        Args:
            encrypted_data: Encrypted bytes from database
            output_path: Where to save decrypted file
        """
        if not self.is_encryption_enabled():
            raise ValueError("Cannot decrypt data - compliance mode not enabled or user not authenticated")
            
        try:
            decrypted_data = self.decrypt_output(encrypted_data, "bytes")
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(decrypted_data)
                
            message("debug", f"Decrypted and saved file: {output_path}")
            
        except Exception as e:
            message("error", f"Failed to decrypt and save file {output_path}: {e}")
            raise

    def calculate_content_hash(self, data: Union[str, bytes]) -> str:
        """
        Calculate SHA256 hash of content for integrity verification.
        
        Args:
            data: Data to hash
            
        Returns:
            Hex string of SHA256 hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()

    def verify_content_hash(self, data: Union[str, bytes], expected_hash: str) -> bool:
        """
        Verify data integrity using hash.
        
        Args:
            data: Data to verify
            expected_hash: Expected SHA256 hash
            
        Returns:
            True if hash matches, False otherwise
        """
        actual_hash = self.calculate_content_hash(data)
        return actual_hash == expected_hash


def should_encrypt_outputs() -> bool:
    """
    Check if outputs should be encrypted (compliance mode + authenticated).
    
    Returns:
        True if outputs should be encrypted, False for normal filesystem storage
    """
    return EncryptionManager().is_encryption_enabled()


def get_encryption_manager() -> EncryptionManager:
    """Get singleton EncryptionManager instance."""
    if not hasattr(get_encryption_manager, "_instance"):
        get_encryption_manager._instance = EncryptionManager()
    return get_encryption_manager._instance


# Output type definitions for database storage
class OutputType:
    """Constants for different output types that can be encrypted."""
    
    # High priority - contains sensitive subject data
    DATABASE = "database"           # SQLite database file
    METADATA_JSON = "metadata_json" # JSON metadata files
    PROCESSING_LOG = "processing_log" # CSV processing logs
    ACCESS_LOG = "access_log"       # Compliance audit logs
    
    # Medium priority - processing details and QC metrics  
    REPORT_PDF = "report_pdf"       # PDF processing reports
    APPLICATION_LOG = "app_log"     # Application log files
    BAD_CHANNELS = "bad_channels"   # TSV bad channels files
    
    # Lower priority - visualization files
    PLOT_PNG = "plot_png"           # PNG visualization files
    PLOT_PDF = "plot_pdf"           # PDF visualization reports
    ICA_REPORT = "ica_report"       # ICA component reports
    
    # Configuration and authentication
    USER_CONFIG = "user_config"     # User configuration files
    AUTH_CONFIG = "auth_config"     # Authentication configuration


def get_output_priority(output_type: str) -> int:
    """
    Get encryption priority for output type.
    
    Args:
        output_type: Output type constant
        
    Returns:
        Priority level (1=highest, 3=lowest)
    """
    high_priority = [
        OutputType.DATABASE,
        OutputType.METADATA_JSON, 
        OutputType.PROCESSING_LOG,
        OutputType.ACCESS_LOG
    ]
    
    medium_priority = [
        OutputType.REPORT_PDF,
        OutputType.APPLICATION_LOG,
        OutputType.BAD_CHANNELS
    ]
    
    if output_type in high_priority:
        return 1
    elif output_type in medium_priority:
        return 2
    else:
        return 3


def encrypt_and_store_output(
    run_id: str,
    output_type: str,
    file_path: Path,
    original_data: Optional[Union[str, bytes, Dict]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    compress: bool = True
) -> Optional[int]:
    """
    High-level function to encrypt and store output in database.
    
    Args:
        run_id: Run ID this output belongs to
        output_type: Type of output (use OutputType constants)
        file_path: Path to file to encrypt (or where it would be saved)
        original_data: Data to encrypt (if not reading from file_path)
        metadata: Additional metadata to store
        compress: Whether to compress before encryption
        
    Returns:
        Database ID of stored output if encrypted, None if normal mode
    """
    encryption_manager = get_encryption_manager()
    
    if not encryption_manager.is_encryption_enabled():
        # Normal mode - no encryption needed
        return None
    
    try:
        # Get data to encrypt
        if original_data is not None:
            if isinstance(original_data, dict):
                # Convert dict to JSON string for consistent hashing
                data_to_encrypt = original_data
                data_for_hash = json.dumps(original_data, sort_keys=True, ensure_ascii=False).encode('utf-8')
                original_size = len(data_for_hash)
            elif isinstance(original_data, str):
                data_to_encrypt = original_data
                data_for_hash = original_data.encode('utf-8')
                original_size = len(data_for_hash)
            else:
                # Assume bytes
                data_to_encrypt = original_data
                data_for_hash = original_data
                original_size = len(original_data)
        elif file_path.exists():
            with open(file_path, "rb") as f:
                data_to_encrypt = f.read()
                data_for_hash = data_to_encrypt
                original_size = len(data_to_encrypt)
        else:
            raise FileNotFoundError(f"File not found and no data provided: {file_path}")
        
        # Calculate content hash before encryption using consistent data format
        content_hash = encryption_manager.calculate_content_hash(data_for_hash)
        
        # Encrypt the data
        encrypted_data = encryption_manager.encrypt_output(data_to_encrypt, compress=compress)
        
        if encrypted_data is None:
            return None
        
        # Store in database
        from autoclean.utils.database import store_encrypted_output
        
        output_id = store_encrypted_output(
            run_id=run_id,
            output_type=output_type,
            file_name=file_path.name,
            encrypted_data=encrypted_data,
            file_size=len(encrypted_data),
            content_hash=content_hash,
            original_path=str(file_path),
            original_size=original_size,
            compression_used=compress,
            metadata=metadata
        )
        
        message("info", f"Encrypted and stored {output_type}: {file_path.name}")
        return output_id
        
    except Exception as e:
        message("error", f"Failed to encrypt and store {output_type}: {e}")
        raise


def decrypt_and_export_output(output_id: int, export_path: Path) -> bool:
    """
    High-level function to decrypt and export output from database.
    
    Args:
        output_id: Database ID of encrypted output
        export_path: Where to save decrypted file
        
    Returns:
        True if successful, False otherwise
    """
    encryption_manager = get_encryption_manager()
    
    if not encryption_manager.is_encryption_enabled():
        raise ValueError("Cannot decrypt - compliance mode not enabled or user not authenticated")
    
    try:
        # Get encrypted data from database
        from autoclean.utils.database import get_encrypted_output_data
        
        output_data = get_encrypted_output_data(output_id)
        encrypted_data = output_data["encrypted_data"]
        content_hash = output_data["content_hash"]
        
        # Decrypt and save
        decrypted_data = encryption_manager.decrypt_output(encrypted_data, "bytes")
        
        # Verify integrity
        if not encryption_manager.verify_content_hash(decrypted_data, content_hash):
            raise ValueError("Data integrity check failed - content hash mismatch")
        
        # Save to file
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "wb") as f:
            f.write(decrypted_data)
        
        message("info", f"Decrypted and exported to: {export_path}")
        return True
        
    except Exception as e:
        message("error", f"Failed to decrypt and export output {output_id}: {e}")
        return False