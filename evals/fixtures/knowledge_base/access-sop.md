# Production Database Access SOP

This standard operating procedure describes exactly how an engineer
requests, receives, and loses production database access. It exists
separately from the Security Policy's general statement of the rule so that
the operational steps can change without a full security policy revision
every time a tool or form changes.

This is version one of the SOP, covering the initial rollout of the access
request form.

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

Approval requires sign-off from the data owner for the system in question
and from one engineering manager. Both approvals must be recorded in the
form before access is provisioned; a verbal or chat approval is not
sufficient on its own and must be followed by the approver clicking approve
in the form.

## Duration and expiration

Once approved, access is granted for seven days and expires automatically
at the end of that window. There is no manual step required to revoke it;
the provisioning system removes the grant on a schedule. An engineer whose
work is not finished within seven days must submit a new request rather
than ask for an extension, so that every active grant in the system
reflects a request that was actually reviewed within the last week.

## Break-glass emergency access

During an incident, an on-call engineer may request break-glass access
without waiting for standard approval, following the break-glass procedure
described in the Security Policy. Break-glass access is logged
automatically and reviewed by a security engineer the next business day.

## Logging and audit

Every access grant, whether standard or break-glass, is logged with the
requesting engineer's identity, the approver's identity, the system
accessed, and the timestamps of grant and expiration. This log is retained
according to the Data Retention Policy and is reviewed monthly by Security
Engineering as part of the standard access review.

## Revoking access early

A manager or the data owner may revoke an engineer's access before the
seven-day window expires if the underlying need has ended or if there is a
concern about how the access is being used. Early revocation does not
require the same two-approver process that granting access does; either
approver alone can revoke.

## Common mistakes to avoid

The most common reason a request is sent back for revision is a vague
reason field, such as "need access for debugging" with no system named.
Requests should name the specific database and the specific reason. The
second most common mistake is requesting broader access than the task
requires, such as write access for a task that only needs to read data;
reviewers are instructed to push back on over-broad requests rather than
approve them for convenience.

## Auditing an existing grant

Any engineer, manager, or security team member can look up the current
status of an access grant through the access request form's history view,
which shows who requested it, who approved it, when it was granted, and
when it will expire. This view does not require special permission beyond
normal system access, since visibility into who has access to what is
itself a security control, not something that needs to be separately
gated.

## Requesting access for automation

Service accounts and automated jobs that need production database access
follow a separate, more restrictive process than the one described above
for individual engineers. A service account request requires sign-off from
the data owner and from the engineering manager of the team that owns the
automation, and access does not expire automatically after seven days the
way an individual grant does, since an automated job may need standing
access. Instead, service account access is reviewed quarterly alongside
the vendor access review described in the Vendor Access Policy.

## Relationship to the Security Policy

This SOP implements the general rule stated in the Security Policy's
production database access section. Where this SOP and the Security Policy
appear to disagree on an operational detail, this SOP's more specific and
more recently updated text governs, since the Security Policy explicitly
defers the operational steps to this document. Any change to the
substance of the rule, such as who must approve a request, is still
expected to be reflected in both documents rather than only here.

## Training prerequisite

Engineers requesting standard production access for the first time must
have completed the security awareness training described in the Security
Policy before their first request will be approved. This is checked
manually by the approving manager in the current version of the form; a
future version of this SOP is expected to automate the check.

## Requesting access to a superseded or archived system

Some databases are archived rather than actively serving production
traffic but still contain data subject to a legal hold or an ongoing
investigation, as described in the Data Retention Policy. Access to an
archived database still follows this SOP's standard approval process; it
is not treated as lower-risk merely because the system is no longer
actively serving traffic, since the data it holds can be just as sensitive
as an active system's data.

## Future changes under consideration

Security Engineering has proposed shortening the standard grant window and
adding a second approver for higher-risk systems, similar to changes
already made for some other access types; any such change would be
published as a new version of this SOP with a clear description of what
changed and why, following the same pattern this document itself uses to
describe its own version history.
