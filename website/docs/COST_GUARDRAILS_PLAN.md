# Cost Guardrails Plan (Top Priority: CloudFront + WAF)

## Goal
Prevent runaway spend (e.g., multi-thousand-dollar surprise in a single day) for the public static site.

## Non-goal
Relying on delayed billing alarms as the primary control.

---

## Priority 1 — CloudFront + WAF (route/path-based controls)

### Why this is first
- Works on request traffic in near real time.
- Can block abuse before large egress accumulates.
- Gives path-level control (`/search/` is higher risk than static assets).

### Plan
1. Put CloudFront in front of S3 website content.
2. Attach AWS WAF Web ACL to CloudFront.
3. Add **rate-based rules with scope-down by URI path**:
   - Rule A (strict): `/search*` (low threshold per IP / 5 min)
   - Rule B (medium): `/view*` (moderate threshold)
   - Rule C (global catch-all): all paths (higher threshold)
4. Add allowlist rule for trusted/admin IPs before rate rules.
5. Add managed baseline rules (CommonRuleSet) after explicit allowlist.
6. Enable WAF logging for tuning thresholds.

### Free tier / pricing note
- **CloudFront** may have a free tier allocation (typically first 12 months for new accounts), but do not depend on it for safety.
- **AWS WAF** generally does **not** provide a meaningful free tier for production usage; expect fixed + request-based charges.
- Action: verify current pricing/free-tier on AWS pricing pages before launch.

---

## Priority 2 — Bucket policy limits (force traffic through CloudFront)

### Why
If bucket remains directly public, attackers can bypass CloudFront/WAF and hit S3 directly.

### Plan
1. Make S3 bucket private (block public access enabled).
2. Use CloudFront Origin Access Control (OAC).
3. Bucket policy: allow `s3:GetObject` **only** from CloudFront distribution principal.
4. Deny non-TLS requests (`aws:SecureTransport = false`).

### Result
All public traffic must traverse CloudFront + WAF controls.

---

## Priority 3 — Budgets: alerts vs actual controls

### Key point
- Budget alerts alone are not hard-stop controls.
- Cost/billing data can lag; a same-day spend spike can happen before alert delivery.

### Plan
1. Keep monthly AWS Budget alerts (50/80/100%) for visibility.
2. Add Budget Actions where possible (secondary safeguard), but do not treat as immediate kill switch.
3. Add **near-real-time operational kill switch**:
   - CloudWatch alarms on CloudFront `Requests` and `BytesDownloaded`.
   - Alarm triggers Lambda to:
     - temporarily block all traffic in WAF (except allowlist), **or**
     - disable CloudFront distribution.
4. Define manual emergency runbook:
   - one command to apply emergency WAF block rule,
   - one command to disable distribution,
   - one command to restore service.

---

## Recommended rollout sequence
1. CloudFront distribution + WAF Web ACL + path-based rate rules.
2. Lock S3 bucket to CloudFront OAC only.
3. Add CloudWatch traffic alarms + auto kill switch Lambda.
4. Keep budgets/anomaly alerts as final visibility layer.

---

## Acceptance criteria
- Direct S3 object URL access is denied.
- `/search*` is rate-limited independently from other routes.
- Simulated request burst triggers WAF block behavior.
- High traffic alarm can auto-trigger emergency block/disable workflow.
- Budget alerts still configured but not relied on as primary protection.
