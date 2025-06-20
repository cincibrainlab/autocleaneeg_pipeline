# Database Access Log Security Implementation Plan

## Overview
Convert file-based database access logging to database-stored, tamper-proof audit trail with CLI export functionality.

## Phase 1: Database Table Setup
**Goal**: Create write-only, tamper-proof table for access logs

### Steps:
- [ ] 1.1 Create `database_access_log` table in database schema
  - `log_id` INTEGER PRIMARY KEY AUTOINCREMENT
  - `timestamp` TEXT NOT NULL
  - `operation` TEXT NOT NULL  
  - `user_context` TEXT NOT NULL (JSON)
  - `database_file` TEXT NOT NULL
  - `details` TEXT (JSON, optional)
  - `log_hash` TEXT (for integrity chain)

- [ ] 1.2 Add database triggers for tamper protection
  - CREATE TRIGGER to prevent UPDATE operations on access log table
  - CREATE TRIGGER to prevent DELETE operations on access log table
  - Only allow INSERT operations

- [ ] 1.3 Implement cryptographic integrity chain
  - Each new log entry includes hash of previous entry
  - Creates tamper-evident chain that detects modifications
  - Store chain hash in `log_hash` field

### Test Requirements:
- [ ] Verify INSERT operations work correctly
- [ ] Verify UPDATE operations are blocked by triggers
- [ ] Verify DELETE operations are blocked by triggers
- [ ] Test hash chain integrity detection

## Phase 2: Migrate Access Logging
**Goal**: Replace file-based logging with database storage

### Steps:
- [ ] 2.1 Update `log_database_access()` function
  - Remove file writing logic
  - Add database INSERT logic
  - Implement hash chain calculation
  - Handle serialization of user_context and details

- [ ] 2.2 Remove file-based logging infrastructure
  - Remove daily log file creation
  - Remove access_logs directory creation
  - Clean up old JSONL file handling

- [ ] 2.3 Update database initialization
  - Ensure access log table is created during setup
  - Add triggers during database creation
  - Initialize hash chain with genesis entry

### Test Requirements:
- [ ] Test access log entries are written to database
- [ ] Verify hash chain integrity is maintained
- [ ] Test all existing pipeline operations still log correctly
- [ ] Verify no file-based logs are created

## Phase 3: CLI Export Command
**Goal**: Provide secure export functionality for access logs

### Steps:
- [ ] 3.1 Add CLI command structure
  - Add `export-access-log` subcommand to autoclean-eeg CLI
  - Support output format options (JSON, CSV, human-readable)
  - Add date range filtering options
  - Add operation type filtering

- [ ] 3.2 Implement export logic
  - Query access log table with filters
  - Verify hash chain integrity during export
  - Format output according to specified format
  - Include integrity verification report

- [ ] 3.3 Add export security features
  - Verify database hasn't been tampered with before export
  - Include hash verification in export output
  - Add digital signature option for exported files
  - Support encrypted export for sensitive data

### CLI Interface Design:
```bash
# Basic export
autoclean-eeg export-access-log --output access-log.json

# Filtered export
autoclean-eeg export-access-log --start-date 2025-01-01 --end-date 2025-01-31 --output monthly-log.csv

# Verify integrity
autoclean-eeg export-access-log --verify-only

# Human readable format
autoclean-eeg export-access-log --format human --output log-report.txt
```

### Test Requirements:
- [ ] Test CLI command registration and parsing
- [ ] Test various output formats
- [ ] Test date range filtering
- [ ] Test integrity verification
- [ ] Test error handling for corrupted chains

## Phase 4: Database Migration & Cleanup
**Goal**: Handle existing installations and cleanup

### Steps:
- [ ] 4.1 Handle existing file-based logs
  - Create migration function to import existing JSONL files
  - Convert file logs to database entries
  - Establish hash chain for migrated data
  - Archive old log files

- [ ] 4.2 Database schema migration
  - Add logic to detect and upgrade existing databases
  - Ensure backwards compatibility during transition
  - Handle edge cases with partial migrations

- [ ] 4.3 Documentation updates
  - Update CLAUDE.md with new access log behavior
  - Document CLI export commands
  - Add troubleshooting guide for integrity issues

### Test Requirements:
- [ ] Test migration of existing log files
- [ ] Test database upgrade scenarios
- [ ] Verify backwards compatibility
- [ ] Test clean installation vs upgrade paths

## Phase 5: Integration & Validation
**Goal**: Ensure complete system integration

### Steps:
- [ ] 5.1 End-to-end testing
  - Run complete pipeline with new logging
  - Verify access logs are created correctly
  - Test export functionality with real data
  - Validate hash chain integrity across operations

- [ ] 5.2 Performance validation
  - Measure database performance impact
  - Ensure logging doesn't slow pipeline significantly
  - Test with high-volume operations

- [ ] 5.3 Security validation
  - Attempt to tamper with access log table
  - Verify tamper detection works correctly
  - Test export security features
  - Validate cryptographic integrity

### Success Criteria:
- [ ] All database operations create access log entries
- [ ] Access log table is tamper-proof (no UPDATEs/DELETEs possible)
- [ ] Hash chain detects any tampering attempts
- [ ] CLI export works with various formats and filters
- [ ] Existing installations migrate cleanly
- [ ] Performance impact is minimal
- [ ] Documentation is complete and accurate

## Risk Mitigation
- **Database corruption**: Regular backups with integrity verification
- **Migration failures**: Rollback procedures and validation checks
- **Performance issues**: Implement batch logging and indexing
- **Compatibility**: Extensive testing across Python versions and OS
- **Security gaps**: Regular security review and penetration testing

## Dependencies
- SQLite trigger support (available in all supported versions)
- cryptography library for hash functions (already in use)
- CLI framework integration (existing autoclean-eeg command structure)
- JSON handling for export formats (standard library)