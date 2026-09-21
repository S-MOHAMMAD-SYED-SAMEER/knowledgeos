# Vendor Access Policy

This policy governs how external vendors, contractors, and other third
parties are granted access to company systems. It exists as a companion to
the Security Policy's general access rules because vendor access carries
risks — a party outside the company's own employment and offboarding
processes — that the standard employee access rules do not fully address.

Security Engineering owns this policy and must approve any new vendor
integration that requires access to a production system or to customer
data, in addition to whatever approval the business owner of the vendor
relationship provides.

## Before granting access

No vendor is granted access to any company system until a signed data
processing agreement is in place covering what data the vendor may access,
how it may be used, and how it must be deleted when the engagement ends.
The business owner of the vendor relationship is responsible for making
sure this agreement exists before requesting access on the vendor's
behalf; Security Engineering does not provision access without confirming
the agreement is on file.

A vendor request must name the specific systems and the specific scope of
access needed, exactly as an internal production database access request
would under the Production Database Access SOP. A vendor asking for
broad, unspecified access should be pushed back on in the same way an
overly broad internal request would be.

## Duration and review

Vendor access is granted for the shorter of the length of the engagement or
ninety days, whichever is less, and must be explicitly renewed if the
engagement continues past that point; there is no automatic extension.
Security Engineering reviews all active vendor access quarterly, separate
from the ninety-day renewal cycle, to confirm each grant is still tied to
an active engagement.

Emergency vendor access, such as a vendor's engineer needing production
access to help resolve an active incident, follows the same break-glass
principle described in the Security Policy: access can be granted
immediately during a declared incident, logged automatically, and reviewed
by a security engineer the next business day.

## Offboarding a vendor

When a vendor engagement ends, the business owner is responsible for
notifying Security Engineering so access can be revoked immediately, in the
same way an employee's manager notifies People Operations at departure
under the Employee Handbook's offboarding section. Vendor access that is
not explicitly revoked at engagement end is caught by the quarterly review
at the latest, but relying on the quarterly review rather than prompt
notification is treated as a process failure, not an acceptable fallback.

## Vendor-managed infrastructure

Some vendors operate infrastructure the company relies on but does not
directly administer, such as a payment processor or an email delivery
service. Access to the vendor's own administrative console for such a
service is treated the same as production database access for approval
purposes: it requires the data owner for that integration and one
engineering manager to approve, and the seven-day standard grant window
from the Production Database Access SOP applies unless the vendor's own
tooling only supports longer-lived credentials, in which case the
exception and its business justification must be documented.

## Data handling by vendors

Vendors must not store company data outside the systems and locations
specified in the data processing agreement. A vendor found to be copying
data to an unapproved location is treated as a security incident under the
Incident Response Runbook, not merely a contract issue to raise separately,
because the risk to company and customer data is the same regardless of
whether the party doing the copying is an employee or a vendor.

## Annual vendor security review

Vendors with standing access to production systems or customer data
undergo an annual security review, coordinated by Security Engineering,
covering their own security practices and any incidents they have reported
in the preceding year. A vendor that fails this review has its access
suspended until the identified issues are remediated, regardless of how
critical the vendor's service is to ongoing operations.

## Vendor personnel changes

The business owner of a vendor relationship is responsible for notifying
Security Engineering promptly when the specific individuals working on the
engagement change, since vendor access is generally granted to named
individuals rather than to the vendor organization as a whole. A vendor
replacing a team member without notifying the company is treated as a
process failure similar to a late offboarding notification, described
above.

## Subprocessors

A vendor that itself uses subprocessors — other third parties who will
have access to company or customer data as part of delivering the vendor's
service — must disclose those subprocessors as part of the data processing
agreement described above. Adding a new subprocessor after the agreement
is signed requires the same review the original vendor engagement went
through, not a lighter-weight notification-only process.

## Vendor access tiers

Vendor access is categorized into read-only, standard, and administrative
tiers, mirroring the internal distinction between read and write access
described in the Production Database Access SOP's guidance against
over-broad requests. Administrative-tier vendor access requires sign-off
from a director-level approver in addition to the standard data owner and
engineering manager approval, reflecting the higher risk of that tier.

## Vendor incident notification obligations

Vendor contracts are expected to include a requirement that the vendor
notify the company within a specified window, typically seventy-two hours,
of any security incident on the vendor's side that could affect company or
customer data. A vendor that fails to meet this notification obligation is
flagged in the annual security review described above regardless of
whether the underlying incident itself caused any actual harm.

## Ending a vendor relationship for cause

If a vendor's annual security review or an incident reveals a serious
enough problem, Security Engineering may recommend suspending or ending
the relationship outright rather than waiting for the standard renewal
cycle. This recommendation goes to the business owner and, for
higher-risk vendors, to the same director-level approver who handles
administrative-tier access requests.
