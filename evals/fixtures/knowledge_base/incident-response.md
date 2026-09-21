# Incident Response Runbook

This runbook defines how the company detects, triages, escalates, and
closes out operational and security incidents. It is owned by Security
Engineering jointly with the on-call rotation leads and is tested at least
twice a year through a tabletop exercise that simulates a real incident
from detection through postmortem.

An incident, for the purposes of this runbook, is any event that degrades a
production service, exposes data it should not have exposed, or creates a
credible risk of either. A single failed request is not an incident; a
pattern that suggests a systemic failure is.

## Severity levels

Incidents are classified into three severity levels at declaration time and
may be re-classified as more information becomes available. Severity one
means a complete outage of a customer-facing service or a confirmed data
exposure; it requires immediate all-hands response and an executive
notification within thirty minutes. Severity two means significant
degradation affecting a subset of customers or a suspected but unconfirmed
data exposure; it requires the on-call engineer and one additional
responder, with an update to the incident channel every thirty minutes.
Severity three means a minor issue with a workaround available; it can be
handled by the on-call engineer alone during normal hours.

## Declaring an incident

Any employee who observes a likely incident can declare one by posting in
the incident channel with the word "incident" and a one-line description.
Declaring an incident is never treated as an overreaction, even if it turns
out to be a false alarm; the cost of an undeclared real incident is always
higher than the cost of a false declaration. The on-call engineer for the
affected service is paged automatically the moment an incident is declared
in the relevant channel.

## Roles during an incident

Every severity-one and severity-two incident has an incident commander, who
coordinates the response but does not necessarily do the hands-on technical
work, and one or more responders who do. The incident commander is
responsible for keeping the incident channel updated, deciding when to
escalate further, and deciding when the incident is resolved. For a
severity-one incident, a separate communications lead handles customer and
executive updates so the incident commander is not pulled away from
coordinating the technical response.

## Break-glass access during an incident

Responders who need production access beyond what they normally hold may
request break-glass access as described in the Security Policy and the
Production Database Access SOP. Break-glass access during a declared
incident does not require the standard approval flow; it is granted
immediately and logged automatically, with review happening after the
incident rather than before access is used.

## Communication

Customer-facing communication during a severity-one incident is drafted by
the communications lead and reviewed by the incident commander before
posting. Internal updates in the incident channel should be factual and
timestamped, avoiding speculation about root cause until it is actually
confirmed. A running timeline is maintained in the incident channel for
every severity-one and severity-two incident and is later used to build the
postmortem.

## Resolution and postmortem

An incident is considered resolved when the customer-facing impact has
stopped, even if the underlying root cause has not yet been fully
addressed; a follow-up task is created to track any remaining work. Every
severity-one incident and every severity-two incident that lasted more than
one hour requires a written postmortem within five business days, covering
timeline, root cause, customer impact, and follow-up actions with owners
and due dates.

Postmortems are blameless: the goal is to understand what allowed the
incident to happen and how to prevent a recurrence, not to identify an
individual at fault. Postmortems are stored in the knowledge base and are
searchable by anyone in the company, not just the team that owned the
incident, since a failure mode in one system is often relevant to others.

## Escalation contacts

Each service has a documented on-call rotation, described in the On-Call
Rotation document, with a primary and secondary on-call engineer at all
times. If the primary does not acknowledge a page within ten minutes, the
secondary is paged automatically; if neither acknowledges within twenty
minutes, the incident commander on duty is paged directly.

## Tooling used during an incident

Incidents are declared and tracked in the incident channel, with the
incident bot automatically creating a dedicated thread, paging the on-call
engineer, and starting a timeline document. Responders should keep
technical discussion in the incident thread rather than in direct
messages, so the eventual postmortem has a complete record rather than
gaps where important context lived in a private conversation.

Status page updates for customer-visible incidents are the communications
lead's responsibility during a severity-one incident and the incident
commander's responsibility during a severity-two incident where no
separate communications lead has been assigned. Status updates should be
factual and should avoid committing to a specific resolution time unless
the team is genuinely confident in it.

## Data exposure incidents

An incident that involves actual or suspected exposure of customer or
employee data is escalated immediately to Legal and to the executive team,
regardless of severity level, in addition to following the standard
severity-one process. Legal determines whether the exposure triggers any
regulatory notification requirements, and Security Engineering leads the
technical investigation into scope: exactly what data was exposed, to
whom, and for how long.

Data exposure incidents follow the same postmortem requirement as other
severity-one incidents, but the postmortem itself may be restricted in
distribution if it contains details Legal advises should not be broadly
shared, with a redacted summary made available to the wider company
instead.

## Vendor-caused incidents

An incident caused by a vendor's system or a vendor's action is handled
through the same process as an internally caused incident, with the
vendor's own incident response team looped in as a responder where
possible. The Vendor Access Policy's annual security review takes a
vendor's incident history over the preceding year into account, so
vendor-caused incidents should be tagged as such in the incident tracking
system rather than only described in free text.

## After-hours and holiday coverage

The on-call rotation continues during holidays and company-wide time off,
though teams may arrange additional coverage or a modified rotation for
particularly high-traffic periods, coordinated through the On-Call
Rotation document. An incident declared during a holiday period follows
the exact same severity and escalation rules as any other day; severity
is never downgraded because of the timing.

## Reviewing incident trends

Engineering leadership reviews incident frequency and severity trends
quarterly, looking for patterns across teams rather than only within a
single service. A service that generates a disproportionate share of
severity-one and severity-two incidents relative to its size is flagged
for a dedicated reliability investment, similar to how the On-Call
Rotation document flags services with high after-hours page volume.
