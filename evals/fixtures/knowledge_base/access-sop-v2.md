# Production Database Access SOP

This standard operating procedure describes exactly how an engineer
requests, receives, and loses production database access. It exists
separately from the Security Policy's general statement of the rule so that
the operational steps can change without a full security policy revision
every time a tool or form changes.

This is version two of the SOP. It tightens the approval and duration rules
that shipped in version one, following a security review after an audit
found that seven-day grants were longer than most requests actually needed
and that a single manager approval was not always sufficient scrutiny for
higher-risk systems.

## Who this applies to

This procedure applies to every engineer who needs direct read or write
access to a production database, including engineers on-call for a service,
engineers debugging a customer-reported issue, and engineers running a
one-time data migration. It does not apply to access granted through an
internal admin tool that already enforces its own audit log and approval
flow; those tools are covered by their own documentation.

## Requesting access

Engineers request production database access through the access request
form, available from the internal tools portal. The form asks for the
specific database and table or schema involved, the reason access is
needed, and the expected duration of the work.

Approval now requires sign-off from the data owner for the system in
question **and from two engineering managers**, one of whom must be from a
different team than the requesting engineer. This second, cross-team
approval is the main change in this version and applies to every request
regardless of system sensitivity. All approvals must be recorded in the
form before access is provisioned; a verbal or chat approval is not
sufficient on its own.

## Duration and expiration

Once approved, access is now granted for **three days**, not seven, and
expires automatically at the end of that window. There is no manual step
required to revoke it; the provisioning system removes the grant on a
schedule. An engineer whose work is not finished within three days must
submit a new request rather than ask for an extension, so that every active
grant in the system reflects a request that was actually reviewed within
the last few days.

Requesting engineers must also have completed the annual security training
within the last ninety days; the access request form checks this
automatically and blocks submission if the training has lapsed, redirecting
the engineer to complete it first.

## Break-glass emergency access

During an incident, an on-call engineer may request break-glass access
without waiting for standard approval, following the break-glass procedure
described in the Security Policy. Break-glass access is logged
automatically and reviewed by a security engineer the next business day.
The break-glass path is unchanged in this version — the tightened approval
and duration rules above apply only to standard, non-emergency requests.

## Logging and audit

Every access grant, whether standard or break-glass, is logged with the
requesting engineer's identity, both approvers' identities, the system
accessed, and the timestamps of grant and expiration. This log is retained
according to the Data Retention Policy and is now reviewed **weekly**
rather than monthly by Security Engineering, as part of the standard access
review.

## Revoking access early

A manager or the data owner may revoke an engineer's access before the
three-day window expires if the underlying need has ended or if there is a
concern about how the access is being used. Early revocation does not
require the same two-approver process that granting access does; either
approver alone can revoke.

## Common mistakes to avoid

The most common reason a request is sent back for revision is a vague
reason field, such as "need access for debugging" with no system named.
Requests should name the specific database and the specific reason. Since
this version introduced the second approver requirement, the second most
common mistake has become submitting a request where both approvers are
from the same team as the requesting engineer; the form now flags this but
does not block it automatically, so reviewers are asked to check it by hand
until that validation ships.

## Auditing an existing grant

Any engineer, manager, or security team member can look up the current
status of an access grant through the access request form's history view,
which shows who requested it, both approvers, when it was granted, and
when it will expire. This view does not require special permission beyond
normal system access, since visibility into who has access to what is
itself a security control, not something that needs to be separately
gated. The history view was updated in this version to show both approvers
rather than the single approver field the version one form used.

## Requesting access for automation

Service accounts and automated jobs that need production database access
follow a separate, more restrictive process than the one described above
for individual engineers. A service account request now requires sign-off
from the data owner and from two engineering managers, matching the
individual-request change above, and access does not expire automatically
after three days the way an individual grant does, since an automated job
may need standing access. Instead, service account access is reviewed
quarterly alongside the vendor access review described in the Vendor
Access Policy.

## Relationship to the Security Policy

This SOP implements the general rule stated in the Security Policy's
production database access section. Where this SOP and the Security
Policy appear to disagree on an operational detail, such as grant
duration, this SOP's more specific and more recently updated text governs,
since the Security Policy explicitly defers the operational steps to this
document. The Security Policy document itself still describes the
version-one seven-day duration as of this writing and is scheduled to be
updated to match; until then, this SOP is the authoritative source for the
current duration.

## Training prerequisite

Engineers requesting standard production access must have completed the
security awareness training described in the Security Policy within the
last ninety days, and, as described above, the access request form now
checks this automatically and blocks submission rather than relying on the
approving manager to check it by hand as the version one process did.
