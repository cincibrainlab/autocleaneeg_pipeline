# Auth0 Integration Implementation Plan

## Overview
Implement Auth0-based user authentication for FDA 21 CFR Part 11 compliance mode in AutoClean EEG pipeline.

**Key Principles:**
- Auth0 handles identity only - no PHI data ever sent to Auth0
- One-time setup during `autoclean setup --compliance-mode`
- Seamless authentication for all subsequent operations
- Local database resilience (auth survives database deletion/recreation)

## Phase 1: Foundation & Dependencies ✅ **COMPLETED**
**Goal**: Set up Auth0 SDK and basic configuration management

- [x] Add `auth0-python` and `requests` to pyproject.toml dependencies
- [x] Create `src/autoclean/utils/auth.py` - Auth0 manager class
- [x] Add compliance mode configuration to user settings
- [x] Create database schema for authenticated users
- [x] Add Auth0 config validation utilities

**Test Requirements:**
- Mock Auth0 responses for unit tests
- Test config validation with invalid Auth0 settings
- Verify database schema creation

**Edge Cases:**
- Handle missing Auth0 configuration gracefully
- Network connectivity issues during auth
- Invalid/expired Auth0 application credentials

## Phase 2: Authentication Flow
**Goal**: Implement browser-based OAuth flow for CLI applications

- [ ] Implement device authorization flow (OAuth 2.0 for CLI)
- [ ] Create local HTTP server for OAuth callback handling
- [ ] Add secure token storage (encrypted local file)
- [ ] Implement automatic token refresh logic
- [ ] Add user session state management

**Test Requirements:**
- Mock OAuth server responses
- Test token refresh scenarios
- Test expired token handling
- Network failure during auth flow

**Edge Cases:**
- Browser not available (headless systems)
- Firewall blocking OAuth callback
- Token corruption/tampering
- Concurrent auth attempts

## Phase 3: CLI Commands
**Goal**: Add login/logout commands and protect compliance operations

- [ ] Add `autoclean login` command with browser flow
- [ ] Add `autoclean logout` command (clear tokens)
- [ ] Add `autoclean whoami` command (show current user)
- [ ] Create authentication middleware for protected commands
- [ ] Add compliance mode detection and enforcement

**Test Requirements:**
- CLI command integration tests
- Mock successful/failed login scenarios
- Test command protection enforcement

**Edge Cases:**
- Multiple users on same system
- Switching between compliance/non-compliance modes
- Command execution with expired tokens

## Phase 4: Setup Integration
**Goal**: Integrate Auth0 setup into existing setup wizard

- [ ] Modify `autoclean setup` to offer compliance mode option
- [ ] Create Auth0 application configuration wizard
- [ ] Add Auth0 connection testing and validation
- [ ] Update workspace configuration format for Auth0 settings
- [ ] Add clear compliance mode indicators in CLI output

**Test Requirements:**
- Mock setup wizard interactions
- Test Auth0 configuration validation
- Test setup rollback on configuration failure

**Edge Cases:**
- Partial Auth0 setup (user exits mid-configuration)
- Invalid Auth0 application configuration
- Auth0 service unavailable during setup
- Existing non-compliance workspace conversion

## Phase 5: Audit Integration
**Goal**: Connect Auth0 user identity with existing audit trail system

- [ ] Update `get_user_context()` to include Auth0 user information
- [ ] Modify database access logging to capture authenticated user
- [ ] Add electronic signatures for compliance operations
- [ ] Update audit trail exports with user authentication metadata
- [ ] Add compliance validation checks to processing pipeline

**Test Requirements:**
- Test audit trail with authenticated vs anonymous users
- Verify electronic signature generation and validation
- Test audit export with Auth0 user context

**Edge Cases:**
- Processing with expired authentication
- Audit trail integrity with mixed auth/non-auth entries
- Electronic signature validation failures

## Phase 6: Testing & Documentation
**Goal**: Comprehensive testing and user documentation

- [ ] Create Auth0 test application for CI/CD
- [ ] Add comprehensive unit tests for all auth components
- [ ] Create integration tests for end-to-end compliance flow
- [ ] Update documentation with compliance setup guide
- [ ] Add troubleshooting guide for common Auth0 issues
- [ ] Create compliance validation checklist

**Test Requirements:**
- Full end-to-end compliance mode testing
- Auth0 service disruption testing
- Multi-user scenarios
- Cross-platform compatibility testing

**Edge Cases:**
- Auth0 service outages
- Network connectivity issues
- Platform-specific browser/OAuth issues
- Corporate firewall restrictions

## Implementation Notes

### Auth0 Application Setup
Users will need to create an Auth0 application with:
- Application Type: Native (for CLI)
- Allowed Callback URLs: `http://localhost:8080/callback`
- Allowed Logout URLs: `http://localhost:8080/logout`
- Grant Types: Authorization Code, Refresh Token

### Database Schema Changes
```sql
-- New table for authenticated users
CREATE TABLE authenticated_users (
    auth0_user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    name TEXT,
    last_login TEXT,
    token_expires TEXT
);

-- Add user context to existing tables
ALTER TABLE runs ADD COLUMN auth0_user_id TEXT;
ALTER TABLE database_access_log ADD COLUMN auth0_user_id TEXT;

-- Electronic signatures table
CREATE TABLE electronic_signatures (
    signature_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    auth0_user_id TEXT NOT NULL,
    signature_data TEXT NOT NULL,  -- JSON with signature details
    timestamp TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (auth0_user_id) REFERENCES authenticated_users(auth0_user_id)
);
```

### Configuration Structure
```yaml
# Added to user configuration
compliance:
  enabled: true
  auth_provider: "auth0"
  auth0:
    domain: "your-tenant.auth0.com"
    client_id: "your-client-id"
    client_secret: "your-client-secret"  # encrypted
    audience: "https://your-tenant.auth0.com/api/v2/"
```

### Security Considerations
- Store Auth0 client secret encrypted locally
- Use short-lived access tokens with refresh tokens
- Implement token rotation
- Log all authentication events in audit trail
- Validate Auth0 JWT tokens properly
- Handle Auth0 service outages gracefully

### User Experience Flow
1. `autoclean setup` → Choose compliance mode → Auth0 configuration
2. `autoclean login` → Browser opens → User authenticates → Token stored
3. `autoclean process ...` → Auto-check auth → Process with user context
4. All operations logged with authenticated user identity

This plan provides a structured approach to implementing Auth0 authentication while maintaining the existing audit trail system and ensuring FDA 21 CFR Part 11 compliance requirements are met.