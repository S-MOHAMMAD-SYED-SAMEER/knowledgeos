# Data Retention Policy

This policy defines how long different categories of data are kept and
what happens to data once its retention period ends. It exists so that
retention is a deliberate, documented decision rather than an accident of
whatever a given system's default happens to be. Security Engineering owns
this policy jointly with Legal, since retention periods are often driven by
regulatory requirements rather than by preference.

Every system that stores customer or employee data is expected to
implement the retention period that applies to the category of data it
holds, including automatic deletion once the period ends where this policy
calls for it. A system that cannot enforce its assigned retention period
technically must have a documented manual process instead, reviewed at
least quarterly.

## Categories of data

Customer account data, such as name, email, and billing information, is
retained for as long as the account is active and for seven years after
account closure to satisfy financial record-keeping requirements. Customer
content data, meaning the actual documents or records a customer stores in
the product, is retained for as long as the account is active and is
deleted within ninety days of account closure unless the customer's
contract specifies otherwise.

Employee records, including HR files and performance history, are retained
for the duration of employment and for seven years after departure,
matching common employment-law requirements across the jurisdictions the
company operates in. Security and access logs, including the access grant
logs described in the Production Database Access SOP, are retained for one
year.

## Incident and postmortem records

Postmortems and incident timelines, as described in the Incident Response
Runbook, are retained indefinitely rather than on a fixed schedule, since
they are a primary source of institutional knowledge about past failures
and their value does not diminish the way transactional log data does.

## Backups

Database backups are retained for thirty days on a rolling basis, after
which older backups are automatically deleted. A backup that falls outside
the thirty-day window is not recoverable through the standard restore
process even if the underlying data would otherwise still be within its
retention period; a longer-term archival backup process exists separately
for data with legal hold requirements and is handled case by case with
Legal.

## Legal holds

When Legal issues a hold on a specific customer's or employee's data for
litigation or regulatory reasons, the normal retention and deletion
schedule for that specific data is suspended until the hold is lifted, even
if the data would otherwise have been deleted under this policy in the
meantime. Legal holds are tracked centrally so that an engineer running a
routine deletion job can check whether a hold applies before deleting
anything covered by one.

## Requesting an exception

A team that believes its system needs a different retention period than
this policy specifies, whether shorter for privacy reasons or longer for a
documented business need, submits a request to Security Engineering and
Legal jointly. Exceptions are recorded in this document's appendix, which
is maintained separately so the core policy stays a stable reference rather
than accumulating one-off carve-outs directly in its main text.

## Deletion verification

Systems that implement automatic deletion are expected to log that
deletion occurred, separately from the data itself, so that Security
Engineering can verify during an audit that deletion is actually happening
on schedule rather than merely being described in a policy document. A
system whose deletion logging cannot be verified is treated as
non-compliant even if engineers believe the deletion is working correctly.

## Analytics and aggregated data

Aggregated, de-identified analytics data that cannot be traced back to a
specific customer or employee is not subject to the per-category retention
periods above and may be retained indefinitely for product and business
analysis, provided the de-identification process meets the standard
reviewed by Legal. Data that could be re-identified by combining it with
other available data is treated as identifiable for retention purposes,
not as aggregated data.

## Retention for terminated accounts under investigation

If a customer or employee account is subject to an active investigation,
whether internal or by a regulator, the standard retention and deletion
schedule for that account's data is suspended in the same way a legal hold
suspends it, even without a formal legal hold having been issued yet.
Security Engineering coordinates with Legal to determine when the
investigation-related hold can be lifted and normal retention resumed.

## Data minimization

Systems should collect only the data actually needed for their stated
purpose, since data that was never collected does not need to be retained
or deleted later. New features that would collect a new category of
customer or employee data are expected to specify their intended retention
period as part of the design review, rather than defaulting to indefinite
retention because a period was not considered at design time.

## Retention schedule changes

Changes to the retention periods defined in this policy require sign-off
from both Security Engineering and Legal, and, for customer-facing data
categories, are communicated to customers through the standard product
change notification process if the change shortens how long their data is
kept. Lengthening a retention period for a legitimate business or legal
reason does not require customer notification under current policy, though
Legal may determine otherwise for a specific case.

## Data retention for backups after account deletion

When a customer account is deleted, the primary copy of the customer's
content data is removed within the ninety-day window described above, but
the data may persist in rolling backups for up to the standard thirty-day
backup retention window described above. This means a deleted account's
data can still theoretically exist in a backup for a short overlap period,
which is disclosed in the customer-facing data handling documentation
rather than treated as an exception to this policy.

## Data retention for evaluation and testing

Fixture and test data used for engineering evaluation purposes, including
data explicitly generated for testing rather than derived from a real
customer, is not subject to the retention periods in this policy, since it
was never customer or employee data to begin with. Engineers must not use
real customer data for testing or evaluation without going through the
same data handling review any other use of customer data would require.

## Contractor and vendor data retention

Data a vendor holds on the company's behalf follows the retention period
specified in that vendor's data processing agreement, described in the
Vendor Access Policy, rather than automatically inheriting this policy's
categories. Security Engineering confirms during the annual vendor
security review that the vendor's actual retention practice matches what
the agreement specifies, the same way it verifies deletion logging for
internal systems.

## Retention documentation for auditors

External auditors reviewing the company's data handling practices,
whether for a customer's own compliance requirements or for a regulatory
audit, are given access to this policy and to the deletion verification
logs described above rather than to the underlying customer data itself.
Security Engineering coordinates auditor access requests and confirms
scope with Legal before granting it, following a review process similar in
spirit to the vendor access approval described in the Vendor Access
Policy, though auditors are not vendors and are tracked separately.
