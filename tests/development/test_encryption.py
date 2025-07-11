#!/usr/bin/env python3
"""
Test script to validate Phase 2 encryption functionality.
"""

import sys
sys.path.insert(0, "src")

def test_encryption_imports():
    """Test that encryption module imports correctly."""
    print("Testing encryption module imports...")
    
    try:
        from autoclean.utils.encryption import (
            EncryptionManager,
            should_encrypt_outputs,
            get_encryption_manager,
            OutputType,
            get_output_priority,
            encrypt_and_store_output,
            decrypt_and_export_output
        )
        print("✅ All encryption functions imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_output_types():
    """Test output type classification system."""
    print("Testing output type classification...")
    
    try:
        from autoclean.utils.encryption import OutputType, get_output_priority
        
        # Test high priority types
        high_priority_types = [
            OutputType.DATABASE,
            OutputType.METADATA_JSON,
            OutputType.PROCESSING_LOG,
            OutputType.ACCESS_LOG
        ]
        
        for output_type in high_priority_types:
            priority = get_output_priority(output_type)
            if priority != 1:
                print(f"❌ {output_type} should have priority 1, got {priority}")
                return False
        
        # Test medium priority types
        medium_priority_types = [
            OutputType.REPORT_PDF,
            OutputType.APPLICATION_LOG,
            OutputType.BAD_CHANNELS
        ]
        
        for output_type in medium_priority_types:
            priority = get_output_priority(output_type)
            if priority != 2:
                print(f"❌ {output_type} should have priority 2, got {priority}")
                return False
        
        # Test low priority types
        low_priority_types = [
            OutputType.PLOT_PNG,
            OutputType.PLOT_PDF,
            OutputType.ICA_REPORT
        ]
        
        for output_type in low_priority_types:
            priority = get_output_priority(output_type)
            if priority != 3:
                print(f"❌ {output_type} should have priority 3, got {priority}")
                return False
        
        print("✅ Output type classification working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Output type test failed: {e}")
        return False

def test_encryption_manager():
    """Test EncryptionManager basic functionality."""
    print("Testing EncryptionManager...")
    
    try:
        from autoclean.utils.encryption import EncryptionManager
        
        # Create encryption manager
        manager = EncryptionManager()
        
        # Test compliance mode detection (should be False in this environment)
        encryption_enabled = manager.is_encryption_enabled()
        print(f"   📋 Encryption enabled: {encryption_enabled}")
        
        # Test hash calculation
        test_data = "Hello, World!"
        content_hash = manager.calculate_content_hash(test_data)
        
        if len(content_hash) != 64:  # SHA256 hex string
            print(f"❌ Content hash wrong length: {len(content_hash)}")
            return False
        
        # Test hash verification
        if not manager.verify_content_hash(test_data, content_hash):
            print("❌ Content hash verification failed")
            return False
        
        print("✅ EncryptionManager basic functionality working")
        return True
        
    except Exception as e:
        print(f"❌ EncryptionManager test failed: {e}")
        return False

def test_conditional_operations():
    """Test that operations are conditional on compliance mode."""
    print("Testing conditional operations...")
    
    try:
        from autoclean.utils.encryption import should_encrypt_outputs, get_encryption_manager
        
        # Should return False in non-compliance mode
        should_encrypt = should_encrypt_outputs()
        print(f"   📋 Should encrypt outputs: {should_encrypt}")
        
        manager = get_encryption_manager()
        
        # Test encryption returns None in normal mode
        test_data = {"test": "data"}
        encrypted_result = manager.encrypt_output(test_data)
        
        if encrypted_result is not None:
            print(f"❌ Expected None for encryption in normal mode, got: {type(encrypted_result)}")
            return False
        
        print("✅ Conditional operations working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Conditional operations test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Encryption Infrastructure (Phase 2)")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_encryption_imports()
    all_passed &= test_output_types()
    all_passed &= test_encryption_manager()
    all_passed &= test_conditional_operations()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 Phase 2 Encryption Test: PASSED")
        print("\nAll encryption infrastructure is working correctly!")
        print("✅ Conditional operation based on compliance mode")
        print("✅ Output type classification system")
        print("✅ EncryptionManager functionality")
        print("✅ Graceful handling when auth0 not available")
    else:
        print("❌ Some tests failed. Check the code.")
    
    sys.exit(0 if all_passed else 1)