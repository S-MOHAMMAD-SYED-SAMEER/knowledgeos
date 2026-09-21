# Security Policy

This policy establishes the minimum security requirements for every
employee, contractor, and system in the company. It applies to company
owned devices, personal devices used for company work under the bring your
own device program, and every internal and third-party system that stores
or processes company data.

Security Engineering owns this policy and reviews it at least twice a year
or whenever a material incident reveals a gap. Questions about how this
policy applies to a specific situation should go to the security channel
before proceeding, not after.

## Account and password requirements

Every employee account must be protected by multi-factor authentication.
Passwords must be at least fourteen characters, must not be reused across
systems, and must not be shared with anyone, including a manager or a
member of the IT team. IT will never ask for a password over email, chat,
or phone; a request that appears to do so should be treated as a phishing
attempt and reported immediately.

Password managers are provided to every employee and are the required
method for generating and storing credentials for work systems. Writing
credentials down, storing them in an unencrypted document, or reusing a
personal password for a work account are all violations of this policy.

## Production database access

Production database access requires production database access approval
through the access request form. Approval requires sign-off from both the
data owner for the system in question and one engineering manager.
Approved access is granted for seven days and expires automatically; a
continuing need requires a fresh request rather than an extension, so that
access grants stay auditable and time-bounded rather than accumulating
indefinitely.

Access requests should state the specific reason access is needed and the
specific system or table involved. A request that says only "need access
for debugging" without naming the system will be sent back for more detail
before it is reviewed.

## Break-glass emergency access

During an active incident, an on-call engineer may request break-glass
access to production systems without waiting for the standard two-approver
review, referenced internally as the break-glass procedure. Break-glass
access is logged automatically, posted to the security channel the moment
it is granted, and reviewed by a security engineer the next business day
regardless of whether the incident is still open.

Break-glass access should be used only when waiting for standard approval
would materially worsen the incident. Using break-glass access for
non-emergency convenience is treated as a policy violation even if the
underlying access would otherwise have been approved through the normal
process.

## Device and data handling

Company laptops are encrypted at rest and must not have disk encryption
disabled under any circumstance. Devices must be locked whenever
unattended, even in the office. Confidential data, including customer data
and unreleased product information, must not be copied to personal devices,
personal cloud storage, or personal email accounts.

Vendors and other third parties are granted access to company systems only
after a signed data processing agreement is in place and only to the
specific systems their engagement requires. Vendor access follows the same
approval and expiration rules as internal production access, tracked
through the vendor access policy, and is reviewed quarterly by Security
Engineering to confirm it is still needed.

## Incident reporting

Any suspected security incident — a lost device, a phishing email that was
clicked, unexpected account activity, or anything that feels like it might
be a problem — should be reported immediately to the security channel.
Employees will never be disciplined for reporting a suspected incident in
good faith, even if it turns out to be a false alarm; the only outcome that
is treated as a problem is not reporting one.

Confirmed incidents are handled according to the Incident Response Runbook,
which defines severity levels, escalation paths, and communication
requirements separately from this policy.

## Data retention

Systems that store customer or employee data follow the retention schedule
defined in the Data Retention Policy. This security policy does not itself
set retention periods; it requires that whatever retention schedule is set
elsewhere is actually enforced by the systems that hold the data, including
automatic deletion where the retention policy calls for it.

## Network and infrastructure security

Production infrastructure is segmented from the general corporate network,
and access between the two requires explicit approval rather than being
open by default. Firewall rule changes to production infrastructure require
review by a second engineer before being applied, tracked through the same
change management process used for other infrastructure changes.

Security patches classified as critical are applied to production systems
within forty-eight hours of release. Patches classified as high severity
are applied within two weeks, and lower-severity patches are bundled into
the regular maintenance cycle. A system that cannot be patched within these
windows for operational reasons requires a documented compensating control
and Security Engineering's sign-off.

## Code and secrets management

Secrets, including API keys, database credentials, and signing keys, are
never committed to source control, even temporarily. Secrets are stored in
the company's secrets manager and accessed by services at runtime rather
than baked into configuration files. A secret that is accidentally
committed must be rotated immediately, not just removed from the commit
history, since the exposure already happened the moment it was pushed.

Code changes to production services require review and approval from at
least one other engineer before merging, and changes touching
security-sensitive code paths, such as authentication or authorization
logic, require review from someone on the security-focused reviewers list
in addition to the standard reviewer.

## Third-party software and open source

New third-party libraries and services must be reviewed before being
adopted for production use, covering their security track record, their
own data handling practices, and whether they have access to customer
data. This review is separate from, and in addition to, the vendor access
approval described in the Vendor Access Policy for cases where the
third-party software also requires system access.

## Security awareness training

Every employee completes security awareness training during onboarding, as
referenced in the Onboarding Guide, and again annually thereafter.
Training covers phishing recognition, password hygiene, and the incident
reporting process described above. Completion is tracked centrally, and an
employee whose training has lapsed may have certain access requests
automatically blocked until it is completed, as described in the
Production Database Access SOP's current version.

## Physical security

Office access requires a badge, and badges are deactivated the same day an
employee departs, coordinated with the offboarding process in the Employee
Handbook. Visitors must be signed in and escorted in any area where
screens might display confidential information. Server rooms and any
on-premise infrastructure, where they exist, require a separate, more
restricted badge tier than general office access.
