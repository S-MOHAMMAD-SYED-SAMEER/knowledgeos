# On-Call Rotation

This document describes how on-call rotations are staffed, how a shift
works, and what is expected of an engineer while they are on call. It is
maintained by each team's on-call lead and reviewed by Engineering
leadership twice a year to make sure the rotation size and shift length
are still reasonable as teams grow or shrink.

Every production service with customer-facing impact has a documented
on-call rotation with a primary and a secondary engineer at all times, as
referenced in the Incident Response Runbook's escalation section.

## Shift structure

A standard on-call shift is one week, running from Monday morning to the
following Monday morning, with a handoff meeting or written handoff note
between the outgoing and incoming on-call engineer. Teams smaller than four
engineers may run shorter shifts to reduce how often any individual is on
call, subject to their engineering manager's approval.

Primary and secondary on-call are staffed simultaneously so that a page
which goes unacknowledged by the primary within ten minutes automatically
escalates to the secondary, matching the escalation timing described in the
Incident Response Runbook.

## Compensation

Engineers receive on-call compensation for each week of primary on-call,
paid as a flat stipend regardless of how many incidents actually occur
during the week. Time spent actively responding to a page outside normal
working hours is additionally compensated or offered as time off in lieu,
depending on local employment law, and should be logged in the time
tracking system under the on-call response category.

## What being on call requires

An on-call engineer must be reachable and able to respond within the
page-acknowledgment window at any time during their shift, including
nights and weekends, and must have working access to a laptop and reliable
internet for the duration. An engineer who knows in advance they will be
unreachable for part of their shift, such as during travel, is responsible
for arranging a swap with a teammate well ahead of time rather than at the
last minute.

On-call engineers are expected to have production access appropriate to
their service already provisioned before their shift starts; requesting
production database access for the first time during an active page is a
sign the rotation's access provisioning process needs fixing, not something
an individual engineer should have to work around under pressure.

## Handling a page

When paged, the on-call engineer should acknowledge the page within the ten
minute window, assess whether the situation meets the bar for declaring an
incident as described in the Incident Response Runbook, and either resolve
it directly if it is minor or declare an incident and follow the runbook if
it is not. Not every page rises to the level of a declared incident; a
single transient error that self-resolves does not need a formal
declaration, but a pattern of related pages should prompt one.

## Rotation membership

New engineers join the on-call rotation once they have completed the
relevant service-specific training and shadowed at least one full shift
with an experienced on-call engineer, typically around the ninety-day mark
described in the Onboarding Guide, though the exact timing is at the
team's discretion based on how quickly the new engineer is ready.

Engineers may request a temporary pause from the rotation for personal
circumstances by talking to their engineering manager; rotations are
adjusted to redistribute shifts fairly among the remaining members rather
than simply skipping the paused engineer's turn and shortening everyone
else's.

## Reviewing rotation health

Each team's on-call lead tracks how many pages occur per shift and how
often pages happen outside working hours, reporting this to Engineering
leadership quarterly. A service that consistently generates a high volume
of after-hours pages is flagged for a reliability investigation rather than
treated as a normal cost of running the service.

## Escalation beyond the rotation

If an incident requires expertise outside the primary and secondary
on-call engineer's own knowledge, either of them can pull in a subject
matter expert directly, following the same incident-channel process
described in the Incident Response Runbook, without needing to first
escalate to the incident commander for permission to do so. Pulling in
extra help early is treated as good judgement, not as a sign the on-call
engineer could not handle the situation alone.

## Cross-team on-call

Some services are jointly owned by more than one team, in which case the
on-call rotation is staffed jointly, with membership and shift allocation
agreed between the owning teams' engineering managers. Joint rotations
follow the same shift length, compensation, and escalation rules as a
single-team rotation; the only difference is who is eligible to be
scheduled.

## New service onboarding to the rotation

When a new production service is launched, its owning team is expected to
have an on-call rotation staffed and documented before the service
receives real customer traffic, not retrofitted afterward. The rotation
documentation for a new service follows the same format as this document
and is reviewed by the on-call lead of an existing, similar service as a
sanity check before the service goes live.

## Retiring a service from the rotation

When a service is deprecated or merged into another system, its on-call
rotation is formally retired rather than left staffed with no real pages
expected; an engineer should not remain on a rotation for a service that
no longer needs one, since that quietly inflates how much on-call burden
the team appears to be carrying. Retiring a rotation is recorded in the
same tracking system used for rotation health review, described below.

## Shadow and backup coverage

Newer members of a rotation, even after completing initial training, may
be paired with a shadow — a more experienced rotation member who is not
formally on call but is available to be consulted — for their first few
live shifts. This is distinct from the primary and secondary structure
described above and is a team-level choice rather than a company-wide
requirement, typically phased out once the newer engineer has handled a
few real pages independently.
