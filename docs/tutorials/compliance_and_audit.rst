Compliance and Audit Trail
===========================

AutoClean provides audit-log export and integrity-check tooling for research
environments that need to track processing activity and inspect database-backed
history. These features can support internal audit practices, but the
repository does not by itself certify compliance with any regulatory framework.

🔒 **Key Features**
-------------------

**Integrity-Oriented Database Logging**
   Database operations are logged with integrity-verification support so runs
   can be reviewed later.

**Hash Chain Integrity**
   Each audit log entry includes a cryptographic hash linking it to the previous entry, creating a tamper-evident chain.

**User Context Tracking**
   Every operation is logged with username, hostname, process ID, and timestamp for complete accountability.

**Task File Tracking**
   The system captures and stores the complete source code and hash of task files used for each processing run.

**CLI Export Tools**
   Export audit logs in multiple formats (JSONL, CSV, human-readable) with filtering and integrity verification.

📋 **Compliance Use Cases**
---------------------------

**Research Data Integrity**
   - Track all modifications to research data
   - Verify that results haven't been tampered with
   - Maintain chain of custody for sensitive data

**Internal Audit Preparation**
   - Export logs for internal review
   - Verify that log history remains internally consistent
   - Preserve task and operator context alongside runs

**Institutional Requirements**
   - IRB audit trail requirements
   - Data governance policies
   - Quality assurance documentation

🛠️ **Using Audit Trail Features**
---------------------------------

**Automatic Logging**

Audit logging is enabled by default - no configuration required. Every time you run AutoClean, the system automatically logs:

- Database operations (create, read, update)
- User context (who ran the operation)
- Timestamps and operation details
- Task file information and hashes

**Exporting Audit Logs**

Export complete audit trail for compliance reporting:

.. code-block:: bash

   # Export all logs to JSONL format
   autocleaneeg-pipeline export-access-log --output complete-audit.jsonl

**Export with Date Filtering**

Filter logs for specific time periods:

.. code-block:: bash

   # Export logs for a specific month
   autocleaneeg-pipeline export-access-log --start-date 2025-01-01 --end-date 2025-01-31 --output january-audit.jsonl
   
   # Export recent activity (last week)
   autocleaneeg-pipeline export-access-log --start-date 2025-06-13 --output recent-activity.jsonl

**Export Different Formats**

Choose the best format for your needs:

.. code-block:: bash

   # JSONL format (default) - best for programmatic analysis
   autocleaneeg-pipeline export-access-log --format json --output audit.jsonl
   
   # CSV format - for spreadsheet analysis
   autocleaneeg-pipeline export-access-log --format csv --output audit.csv
   
   # Human-readable report - for manual review
   autocleaneeg-pipeline export-access-log --format human --output audit-report.txt

**Filter by Operation Type**

Export specific types of operations:

.. code-block:: bash

   # Only export data storage operations
   autocleaneeg-pipeline export-access-log --operation "store" --output storage-operations.jsonl
   
   # Only export database creation events
   autocleaneeg-pipeline export-access-log --operation "create_collection" --output database-creation.jsonl

**Verify Database Integrity**

Check that audit logs haven't been tampered with:

.. code-block:: bash

   # Verify integrity without exporting data
   autocleaneeg-pipeline export-access-log --verify-only

This command checks the cryptographic hash chain and reports any integrity issues.

📊 **Understanding Export Formats**
-----------------------------------

**JSONL Format**

The default JSONL (JSON Lines) format includes:

- **Metadata Line**: Export timestamp, database path, integrity status
- **Log Entries**: One JSON object per line with operation details

Example structure:

.. code-block:: json

   {"type": "metadata", "export_timestamp": "2025-06-20T10:30:00", "total_entries": 25, "integrity_status": "valid"}
   {"type": "access_log", "log_id": 1, "operation": "database_initialization", "user_context": {"user": "researcher", "host": "lab-computer"}}

**CSV Format**

Tabular format suitable for Excel or statistical analysis:

- All log entries in rows
- User context and details flattened to columns
- Easy to filter and sort in spreadsheet applications

**Human-Readable Format**

Formatted text report with:

- Summary header with database information
- Detailed entry-by-entry breakdown
- Hash verification status
- Easy to read for manual review

🔍 **Integrity Verification Details**
-------------------------------------

**How Hash Chain Works**

1. Each log entry includes a hash of the previous entry
2. The first entry links to a "genesis" hash
3. Any modification breaks the chain and is detected
4. Verification checks every link in the chain

**What Triggers Integrity Alerts**

- Modified log entries (tampering detected)
- Missing log entries (deletion detected)
- Broken hash chain (data corruption)
- Database file corruption

**Interpreting Verification Results**

.. code-block:: text

   ✓ All 150 access log entries verified successfully
   → Database integrity confirmed
   
   ✗ Found 2 integrity issues
   → Entry 45: Hash mismatch detected
   → Entry 67: Missing previous hash

Best Practices
--------------

**Regular Exports**

- Export audit logs monthly for ongoing compliance
- Store exports in secure, separate location
- Include exports in backup procedures

**Integrity Verification**

- Run integrity checks before important reports
- Verify integrity after any system maintenance
- Document verification results

**Documentation**

- Include audit trail exports in research documentation
- Reference specific log entries in compliance reports
- Maintain chain of custody documentation

**Access Control**

- Limit database file access to authorized users
- Use file system permissions to protect audit data
- Consider database encryption for sensitive environments

Security Considerations
-----------------------

**Database Protection**

The audit system includes multiple protection layers:

- **SQL Triggers**: Prevent modification of audit records
- **Write-Only Table**: Only allows INSERT operations
- **Hash Chain**: Detects any tampering attempts
- **Automatic Backups**: Regular database backups with integrity checks

**Limitations**

While the audit system is robust, consider these limitations:

- **File System Access**: Users with database file access could replace entire database
- **Root/Admin Access**: System administrators can override protections
- **Backup Integrity**: Backups should be stored securely and verified

For maximum security in regulated environments, consider additional measures like database encryption, file system monitoring, and secure backup storage.

Framework Boundary
------------------

AutoClean can provide evidence that may be useful inside a broader quality
system, but maintainers do not represent the repository as independently
validated for FDA, GCP, GLP, ALCOA+, or similar frameworks. Teams with those
requirements should review the actual outputs, environment controls, backup
policy, access model, and institutional procedures in their own deployment.

Next Steps
----------

**For Compliance Officers**

- Review export formats and determine best fit for your requirements
- Establish regular audit log export procedures
- Integrate verification steps into quality assurance processes

**For Researchers**

- Include audit trail exports in research documentation
- Use integrity verification before publishing results
- Consider audit requirements when planning studies

**For IT Administrators**

- Review security considerations for your environment
- Implement appropriate database protection measures
- Set up automated backup and verification procedures

For more advanced compliance features and customization options, see the :doc:`creating_custom_task` and API reference documentation.
