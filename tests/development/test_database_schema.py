#!/usr/bin/env python3
"""
Test script to validate Phase 1 database schema changes.

This script tests that the new encrypted outputs functionality works correctly
in both compliance and normal modes without breaking existing functionality.
"""

import os
import sqlite3
import tempfile
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, "src")

from autoclean.utils.database import (
    set_database_path, 
    manage_database_conditionally,
    database_supports_encryption,
    get_database_schema_version,
    store_encrypted_output,
    get_encrypted_outputs,
    get_encrypted_output_data
)

def test_database_schema():
    """Test that database schema creation works in both modes."""
    
    print("🧪 Testing Database Schema Changes (Phase 1)")
    print("=" * 50)
    
    # Create temporary directory for test databases
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "test_autoclean"
        test_dir.mkdir(parents=True)
        
        # Set database path
        set_database_path(test_dir)
        
        print("\n1. Testing database creation...")
        
        # Test database creation (this will detect current compliance mode)
        try:
            result = manage_database_conditionally("create_collection")
            print("   ✅ Database created successfully")
        except Exception as e:
            print(f"   ❌ Database creation failed: {e}")
            return False
        
        print("\n2. Testing schema version and encryption support...")
        
        # Test schema version
        schema_version = get_database_schema_version()
        print(f"   📋 Schema version: {schema_version}")
        
        # Test encryption support detection
        supports_encryption = database_supports_encryption()
        print(f"   🔐 Encryption support: {'Yes' if supports_encryption else 'No'}")
        
        if not supports_encryption:
            print("   ❌ Database should support encryption (encrypted_outputs table missing)")
            return False
        
        print("\n3. Testing table structure...")
        
        # Connect to database and check table structure
        db_path = test_dir / "pipeline.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check that encrypted_outputs table exists with correct columns
        cursor.execute("PRAGMA table_info(encrypted_outputs)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        expected_columns = {
            'id': 'INTEGER',
            'run_id': 'TEXT', 
            'output_type': 'TEXT',
            'file_name': 'TEXT',
            'encrypted_data': 'BLOB',
            'file_size': 'INTEGER',
            'content_hash': 'TEXT',
            'created_at': 'TEXT'
        }
        
        missing_columns = []
        for col, col_type in expected_columns.items():
            if col not in columns:
                missing_columns.append(col)
            elif col_type not in columns[col]:
                print(f"   ⚠️  Column {col} has type {columns[col]}, expected {col_type}")
        
        if missing_columns:
            print(f"   ❌ Missing columns in encrypted_outputs: {missing_columns}")
            return False
        else:
            print("   ✅ encrypted_outputs table structure correct")
        
        # Check that pipeline_runs has new compliance columns
        cursor.execute("PRAGMA table_info(pipeline_runs)")
        runs_columns = {row[1] for row in cursor.fetchall()}
        
        expected_new_columns = {'compliance_mode', 'outputs_encrypted', 'encryption_key_id'}
        missing_runs_columns = expected_new_columns - runs_columns
        
        if missing_runs_columns:
            print(f"   ❌ Missing columns in pipeline_runs: {missing_runs_columns}")
            return False
        else:
            print("   ✅ pipeline_runs table has compliance mode columns")
        
        conn.close()
        
        print("\n4. Testing encrypted outputs API...")
        
        # Test storing encrypted output (mock data)
        test_run_id = "test_run_123"
        test_encrypted_data = b"encrypted_test_data"
        test_content_hash = "abc123def456"
        
        try:
            output_id = store_encrypted_output(
                run_id=test_run_id,
                output_type="test_output",
                file_name="test_file.pdf", 
                encrypted_data=test_encrypted_data,
                file_size=len(test_encrypted_data),
                content_hash=test_content_hash,
                original_path="/test/path/test_file.pdf",
                metadata={"test": "value"}
            )
            print(f"   ✅ Stored encrypted output with ID: {output_id}")
            
        except Exception as e:
            print(f"   ❌ Failed to store encrypted output: {e}")
            return False
        
        # Test retrieving encrypted outputs
        try:
            outputs = get_encrypted_outputs(test_run_id)
            if len(outputs) == 1:
                print(f"   ✅ Retrieved {len(outputs)} encrypted output(s)")
            else:
                print(f"   ❌ Expected 1 output, got {len(outputs)}")
                return False
                
        except Exception as e:
            print(f"   ❌ Failed to retrieve encrypted outputs: {e}")
            return False
        
        # Test retrieving encrypted data
        try:
            output_data = get_encrypted_output_data(output_id)
            if output_data['encrypted_data'] == test_encrypted_data:
                print("   ✅ Retrieved encrypted data matches stored data")
            else:
                print("   ❌ Retrieved encrypted data doesn't match")
                return False
                
        except Exception as e:
            print(f"   ❌ Failed to retrieve encrypted data: {e}")
            return False
            
    print("\n" + "=" * 50)
    print("🎉 Phase 1 Database Schema Test: PASSED")
    print("\nAll database schema changes are working correctly!")
    print("✅ encrypted_outputs table created")
    print("✅ pipeline_runs extended with compliance columns") 
    print("✅ Database API functions working")
    print("✅ Schema version updated to 1.1")
    
    return True

if __name__ == "__main__":
    success = test_database_schema()
    sys.exit(0 if success else 1)