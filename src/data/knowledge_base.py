"""
Knowledge Base - Source Documents for RAG Pipeline.

This module provides the source documents that will be:
1. Embedded using Sentence Transformers
2. Indexed in the FAISS vector database
3. Retrieved during similarity search for RAG

In a production system, these documents would come from:
- A CMS or document management system
- A database of support articles
- External API endpoints
- File storage (PDF, Markdown, etc.)

For this implementation, we use synthetic but realistic domain
registrar support documentation (similar to Tucows/OpenSRS).

The documents are loaded once at startup and indexed in FAISS.
"""

from typing import List

from src.models.schemas import Document


def get_knowledge_base() -> List[Document]:
    """
    Returns the synthetic knowledge base documents.
    
    Categories include:
    - Domain Policies
    - WHOIS Information
    - Billing & Payments
    - DNS & Technical
    - Transfer Policies
    - Security & Abuse
    """
    
    return [
        # ============================================
        # DOMAIN SUSPENSION POLICIES
        # ============================================
        Document(
            id="policy-001",
            title="Domain Suspension Guidelines",
            category="Domain Policies",
            section="Section 4.1 - Suspension Reasons",
            content="""
Domain Suspension Guidelines - Section 4.1: Reasons for Suspension

A domain may be suspended for the following reasons:
1. WHOIS Verification Failure: If the registrant fails to verify their email within 15 days of registration or update
2. Invalid WHOIS Information: Providing false, inaccurate, or incomplete contact information
3. Non-Payment: Failure to pay renewal fees before the grace period expires
4. Policy Violation: Violation of the Acceptable Use Policy
5. Legal Compliance: Court orders, UDRP decisions, or law enforcement requests
6. Abuse Reports: Confirmed malware, phishing, or spam activities

Domains suspended for WHOIS issues can be reactivated within 30 days by updating and verifying contact information.
For policy violations, the domain holder must contact the Abuse Team for review.
            """
        ),
        Document(
            id="policy-002",
            title="Domain Suspension Guidelines",
            category="Domain Policies",
            section="Section 4.2 - Reactivation Process",
            content="""
Domain Suspension Guidelines - Section 4.2: Domain Reactivation Process

To reactivate a suspended domain:
1. Log in to your domain management portal
2. Navigate to "My Domains" and select the suspended domain
3. Click "View Suspension Details" to see the reason
4. Complete the required action based on suspension type:
   - WHOIS Issues: Update contact information and click "Resend Verification Email"
   - Payment Issues: Process outstanding payment in the Billing section
   - Policy Violations: Submit an appeal through the Abuse Team portal

Reactivation Timeline:
- WHOIS verification: Usually within 24-48 hours after verification
- Payment: Immediate upon successful payment processing
- Abuse/Policy: 3-5 business days for review

Note: Domains may be permanently deleted if not reactivated within 30 days of suspension.
            """
        ),
        Document(
            id="policy-003",
            title="Domain Suspension Notifications",
            category="Domain Policies",
            section="Section 4.3 - Communication",
            content="""
Domain Suspension Guidelines - Section 4.3: Suspension Notifications

Notification Policy:
- First Warning: 7 days before potential suspension (for WHOIS/payment issues)
- Final Warning: 24 hours before suspension
- Suspension Notice: Immediately upon suspension

Notifications are sent to:
1. Primary registrant email on file
2. Admin contact email (if different)
3. Emergency contact email (if configured)

If you did not receive notifications:
- Check your spam/junk folder
- Verify your email address is current in the domain settings
- Add our notification domain to your safe sender list: @notices.domainregistry.com

To update notification preferences, log in to your account and navigate to "Account Settings" > "Notifications"
            """
        ),
        
        # ============================================
        # WHOIS INFORMATION
        # ============================================
        Document(
            id="whois-001",
            title="WHOIS Information Requirements",
            category="WHOIS Information",
            section="Section 2.1 - Required Fields",
            content="""
WHOIS Information Requirements - Section 2.1: Required Fields

ICANN requires accurate WHOIS information for all domain registrations:

Required Registrant Information:
- Full legal name (individual or organization)
- Valid physical address (P.O. boxes not accepted as primary)
- Working phone number
- Valid email address (must be verified)

Required Admin/Technical Contact:
- Full name of responsible person
- Contact phone number
- Contact email address

Validation Requirements:
- Email addresses must be verified within 15 days
- Phone numbers may be validated via callback
- Address verification may be required for high-risk domains

Consequences of Invalid Information:
- Domain suspension within 15 days of failed verification
- Potential domain cancellation after 30 days
- Loss of dispute resolution rights
            """
        ),
        Document(
            id="whois-002",
            title="WHOIS Privacy Protection",
            category="WHOIS Information",
            section="Section 2.3 - Privacy Services",
            content="""
WHOIS Privacy Protection - Section 2.3: Privacy Services

WHOIS Privacy (Domain Privacy Protection) replaces your personal information in public WHOIS records with proxy contact information.

Benefits:
- Prevents spam and unwanted solicitation
- Reduces risk of identity theft
- Protects personal address from public view

Limitations:
- Does not protect from legitimate legal requests
- May be disabled for certain TLDs (e.g., .us, .ca for individuals)
- Does not exempt you from providing accurate underlying information

Pricing: Included free with domain registration

To enable/disable privacy:
1. Log in to domain management portal
2. Select domain > "WHOIS Settings"
3. Toggle "Privacy Protection" on/off

Note: Disabling privacy may take 24-48 hours to reflect in public WHOIS.
            """
        ),
        
        # ============================================
        # BILLING & PAYMENTS
        # ============================================
        Document(
            id="billing-001",
            title="Domain Renewal Policies",
            category="Billing & Payments",
            section="Section 5.1 - Renewal Process",
            content="""
Domain Renewal Policies - Section 5.1: Renewal Process

Auto-Renewal:
- Domains are set to auto-renew by default
- Renewal is attempted 30 days before expiration
- If payment fails, retries occur at 15 days and 7 days before expiration

Manual Renewal:
- Can be done anytime from 1-10 years in advance
- Navigate to "My Domains" > "Renew Domain"
- Multi-year discounts available for 2+ year renewals

Renewal Pricing:
- Standard renewal: At current market rate
- Early renewal (60+ days before): 5% discount
- Multi-year: Up to 15% discount for 5+ years

Payment Methods Accepted:
- Credit/Debit cards (Visa, MasterCard, Amex)
- PayPal
- Account credit balance
- Wire transfer (for orders over $1000)
            """
        ),
        Document(
            id="billing-002",
            title="Expired Domain Recovery",
            category="Billing & Payments",
            section="Section 5.2 - Grace Periods",
            content="""
Expired Domain Recovery - Section 5.2: Grace Periods

After domain expiration, the following grace periods apply:

1. Renewal Grace Period (0-30 days after expiration)
   - Domain is inactive but can be renewed at standard price
   - Website and email stop working
   - Renew through normal process

2. Redemption Period (31-60 days after expiration)
   - Domain can only be recovered with redemption fee
   - Redemption fee: $80 + renewal fee
   - Must contact support to initiate redemption

3. Pending Delete (61-66 days after expiration)
   - Domain cannot be recovered
   - Will be released to public registration

Important: Premium domains may have different grace periods. Check your domain status page for specific dates.

To recover an expired domain:
1. Log in to your account
2. Go to "Expired Domains"
3. Select domain and click "Restore" or "Redeem"
            """
        ),
        Document(
            id="billing-003",
            title="Refund Policy",
            category="Billing & Payments",
            section="Section 5.4 - Refunds",
            content="""
Refund Policy - Section 5.4: Domain Refunds

Domain Registration Refunds:
- New registrations: Full refund within 5 days of registration
- After 5 days: No refund available (ICANN policy)

Domain Renewal Refunds:
- Within 5 days of renewal: Full refund
- After 5 days: No refund available

Exceptions (No Refunds):
- Domains transferred in
- Premium/aftermarket domains
- Domains with active disputes
- Domains flagged for abuse

To request a refund:
1. Open a support ticket within the refund window
2. Include domain name and order number
3. Provide reason for refund request

Processing time: 5-7 business days for credit cards, 10-14 days for PayPal
            """
        ),
        
        # ============================================
        # DNS & TECHNICAL
        # ============================================
        Document(
            id="dns-001",
            title="DNS Configuration Guide",
            category="DNS & Technical",
            section="Section 3.1 - Nameserver Setup",
            content="""
DNS Configuration Guide - Section 3.1: Nameserver Setup

Default Nameservers:
- ns1.domainregistry.com
- ns2.domainregistry.com

To use custom nameservers:
1. Go to "My Domains" > Select Domain > "Nameservers"
2. Choose "Use custom nameservers"
3. Enter at least 2 nameserver hostnames
4. Save changes

Propagation Time:
- Changes typically propagate within 24-48 hours globally
- Some ISPs may cache old records longer

Common Issues:
- "Lame delegation": Nameservers don't respond - verify NS records at provider
- "Mismatch": NS records point to servers that don't have your zone
- DNSSEC errors: Disable DNSSEC before changing nameservers

Best Practices:
- Always have at least 2 nameservers on different networks
- Test DNS resolution before making changes
- Keep old nameservers active for 48 hours after switch
            """
        ),
        Document(
            id="dns-002",
            title="DNS Record Management",
            category="DNS & Technical",
            section="Section 3.2 - Record Types",
            content="""
DNS Record Management - Section 3.2: Common Record Types

A Record (Address):
- Maps domain to IPv4 address
- Example: yourdomain.com → 192.168.1.1

AAAA Record (IPv6 Address):
- Maps domain to IPv6 address
- Example: yourdomain.com → 2001:db8::1

CNAME Record (Canonical Name):
- Alias one domain to another
- Example: www.yourdomain.com → yourdomain.com
- Cannot be used on root domain (use A record)

MX Record (Mail Exchange):
- Specifies mail servers for the domain
- Requires priority value (lower = higher priority)
- Example: 10 mail.yourdomain.com

TXT Record:
- Stores text data (SPF, DKIM, domain verification)
- Example: v=spf1 include:_spf.google.com ~all

TTL (Time to Live):
- How long DNS records are cached
- Default: 3600 seconds (1 hour)
- Lower TTL = faster propagation but more DNS queries
            """
        ),
        
        # ============================================
        # DOMAIN TRANSFERS
        # ============================================
        Document(
            id="transfer-001",
            title="Domain Transfer Policy",
            category="Transfer Policies",
            section="Section 6.1 - Transfer Requirements",
            content="""
Domain Transfer Policy - Section 6.1: Transfer Requirements

Requirements for transferring a domain TO us:
1. Domain must be unlocked at current registrar
2. Must have valid authorization code (EPP/Auth code)
3. Domain must be older than 60 days
4. Domain must not be within 60 days of expiration
5. Admin contact email must be accessible

Transfer Timeline:
- Standard transfer: 5-7 days
- Fast transfer (some TLDs): 24 hours with expedite fee

Transfer Cost:
- Standard TLDs: One year registration added to domain
- Premium TLDs: Variable pricing, check before initiating

To initiate incoming transfer:
1. Go to "Transfer Domain"
2. Enter domain name
3. Enter authorization code
4. Complete payment
5. Confirm transfer via email
            """
        ),
        Document(
            id="transfer-002",
            title="Domain Transfer Away",
            category="Transfer Policies",
            section="Section 6.2 - Outgoing Transfers",
            content="""
Domain Transfer Policy - Section 6.2: Outgoing Transfers

To transfer your domain away:
1. Ensure domain is unlocked: "Domain Settings" > "Lock Status"
2. Obtain authorization code: "Domain Settings" > "Get Auth Code"
3. Auth code is emailed to registrant email
4. Provide auth code to new registrar

Important Notes:
- We do not charge for outgoing transfers
- Domain must not be within 60 days of registration or previous transfer
- Transfer adds one year to registration (paid to new registrar)
- Ensure WHOIS email is current to receive auth code

Transfer Lock:
- Domains are locked by default to prevent unauthorized transfers
- Unlock required before initiating transfer
- Lock can be re-enabled anytime

If you cannot unlock your domain, contact support with domain verification.
            """
        ),
        
        # ============================================
        # SECURITY & ABUSE
        # ============================================
        Document(
            id="abuse-001",
            title="Abuse Policy",
            category="Security & Abuse",
            section="Section 7.1 - Acceptable Use",
            content="""
Abuse Policy - Section 7.1: Acceptable Use Policy

Prohibited Activities:
1. Phishing: Creating deceptive pages to steal credentials
2. Malware Distribution: Hosting or distributing malicious software
3. Spam: Using domain for unsolicited bulk email
4. Copyright Infringement: Hosting pirated content
5. Illegal Content: Any content violating applicable laws

Abuse Reports:
- Third parties can report abuse at abuse@domainregistry.com
- Reports are reviewed within 24 hours
- Domain holder is notified and given opportunity to respond

Enforcement Actions:
1. Warning email (first offense, minor violation)
2. Temporary suspension (repeated offenses)
3. Permanent suspension (severe violations)
4. Domain termination (criminal activity)

Appeals:
- Submit appeal within 14 days of action
- Include evidence that violation was resolved
- Appeals reviewed within 5 business days
            """
        ),
        Document(
            id="abuse-002",
            title="Domain Security Features",
            category="Security & Abuse",
            section="Section 7.2 - Security Options",
            content="""
Domain Security Features - Section 7.2: Security Options

Registry Lock:
- Highest level of protection against unauthorized changes
- Requires manual verification for any modifications
- Recommended for high-value domains
- Cost: $50/year

DNSSEC (DNS Security Extensions):
- Protects against DNS spoofing attacks
- Available for most TLDs at no cost
- Enable in DNS settings > DNSSEC

Two-Factor Authentication (2FA):
- Add extra security to your account
- Supports authenticator apps and SMS
- Required for Registry Lock domains

Domain Lock:
- Prevents unauthorized transfers
- Enabled by default on all domains
- Can be toggled in domain settings

Security Best Practices:
1. Use strong, unique password for domain account
2. Enable 2FA on your account
3. Keep WHOIS contact emails current and secure
4. Regularly review domain settings and access logs
5. Use Registry Lock for mission-critical domains
            """
        ),
        Document(
            id="abuse-003",
            title="Compromised Domain Recovery",
            category="Security & Abuse",
            section="Section 7.3 - Recovery Process",
            content="""
Compromised Domain Recovery - Section 7.3: Recovery Process

If your domain was hijacked or compromised:

Immediate Steps:
1. Contact support immediately at security@domainregistry.com
2. Include: Domain name, account email, proof of ownership
3. Request emergency domain lock

Verification Process:
- Government-issued ID matching WHOIS registrant
- Utility bill or bank statement matching WHOIS address
- Previous payment receipts or invoices
- Original registration confirmation email

Recovery Timeline:
- Emergency lock: Within 2 hours of verified report
- Investigation: 1-3 business days
- Recovery (if approved): Immediate

Prevention:
- Never share your account credentials
- Use unique, strong passwords
- Enable two-factor authentication
- Regularly review account access logs
- Set up login notifications
            """
        ),
        
        # ============================================
        # ACCOUNT MANAGEMENT
        # ============================================
        Document(
            id="account-001",
            title="Account Access Issues",
            category="Account Management",
            section="Section 1.1 - Login Problems",
            content="""
Account Access Issues - Section 1.1: Login Problems

Forgot Password:
1. Click "Forgot Password" on login page
2. Enter your registered email address
3. Check email for reset link (valid for 24 hours)
4. Set new password (minimum 12 characters, mix of letters/numbers/symbols)

Password Not Working:
- Ensure CAPS LOCK is off
- Try password reset process
- Check if account is locked (5 failed attempts = 15 min lockout)

No Email Access:
1. Contact support with account verification
2. Required: Government ID + domain ownership proof
3. Allow 2-3 business days for verification

Account Locked:
- Automatic unlock after 15 minutes
- Multiple lockouts may require support intervention
- Suspicious activity may require additional verification

Two-Factor Authentication Lost:
- Use backup codes if available
- Contact support with ID verification
- Recovery codes available after identity verification
            """
        ),
        Document(
            id="account-002",
            title="Account Closure",
            category="Account Management",
            section="Section 1.3 - Closing Account",
            content="""
Account Closure - Section 1.3: Closing Your Account

Before closing your account:
1. Transfer or cancel all active domains
2. Download any important records/invoices
3. Ensure no outstanding balance

To close account:
1. Log in to your account
2. Go to "Account Settings" > "Close Account"
3. Confirm closure (requires password)

Important Notes:
- Domains must be transferred or expired before closure
- Account history retained for 7 years (legal requirement)
- Can reopen within 90 days by contacting support
- After 90 days, email address can be reused for new account

Outstanding Balances:
- Accounts with unpaid invoices cannot be closed
- Pay balance or contact billing for arrangements
            """
        ),
        
        # ============================================
        # FAQs
        # ============================================
        Document(
            id="faq-001",
            title="FAQ: Domain Registration",
            category="FAQ",
            section="Registration FAQs",
            content="""
Frequently Asked Questions - Domain Registration

Q: How long can I register a domain for?
A: Domain registrations are available from 1-10 years. Longer terms provide cost savings and protect against price increases.

Q: Can I register any domain name I want?
A: Domain names are first-come, first-served. Some restrictions apply:
   - Cannot match trademarked terms (may be challenged)
   - Some TLDs have eligibility requirements
   - Premium domains may have higher pricing

Q: What happens if someone else has the domain I want?
A: Options include:
   - Set up backorder notification for when it expires
   - Make an offer through our brokerage service
   - Try alternative TLDs (.net, .co, .io, etc.)

Q: How do I protect my brand?
A: We recommend:
   - Register common TLD variations
   - Consider defensive registrations of misspellings
   - Enable domain monitoring for similar registrations
            """
        ),
        Document(
            id="faq-002",
            title="FAQ: Email & Website Issues",
            category="FAQ",
            section="Technical FAQs",
            content="""
Frequently Asked Questions - Email & Website Issues

Q: My website is not working after domain registration. Why?
A: Common causes:
   - DNS propagation: Wait 24-48 hours after initial setup
   - Hosting not configured: Ensure A records point to hosting IP
   - Hosting issue: Contact your hosting provider

Q: My email stopped working. What should I check?
A: Verify:
   - MX records are correctly configured
   - Email provider is active and paid
   - Domain hasn't expired or been suspended
   - No DNS changes were recently made

Q: Why does my website show a parking page?
A: Default parking pages appear when:
   - Domain was just registered and not configured
   - Hosting has lapsed
   - DNS records are not set
   
Resolution: Configure your nameservers or DNS records to point to your hosting.

Q: How do I set up email forwarding?
A: Go to "Domain Settings" > "Email Forwarding" and add forwarding rules. Note: Full email hosting provides more features than forwarding.
            """
        ),
    ]
