# HTTPS Plan (Static Site)

## Goal
Serve the site over HTTPS with modern TLS and no direct public S3 website endpoint exposure.

## Current state
- S3 static website endpoint is HTTP-only.
- Public bucket website hosting is simple but not HTTPS-safe by itself.

---

## Recommended architecture

1. **CloudFront distribution** in front of S3.
2. **S3 origin uses REST endpoint** (not website endpoint).
3. **Origin Access Control (OAC)** so only CloudFront can read bucket objects.
4. **ACM certificate (us-east-1)** attached to CloudFront.
5. Optional custom domain via Route53 alias to CloudFront.

This is the standard AWS path to HTTPS for static sites.

---

## Implementation steps

### Phase 1 — CloudFront over existing content
1. Create CloudFront distribution with S3 origin.
2. Set default root object: `index.html`.
3. Viewer protocol policy: `redirect-to-https`.
4. Minimum TLS policy: modern (e.g., TLSv1.2_2021).
5. Keep existing S3 website endpoint temporarily for rollback.

### Phase 2 — Lock down S3
1. Enable S3 public access block.
2. Add OAC and bucket policy allowing only CloudFront principal.
3. Remove public-read bucket policy.

### Phase 3 — Certificate and domain
1. Request ACM cert in `us-east-1`.
2. Validate cert (DNS validation preferred).
3. Attach cert to CloudFront.
4. (Optional) Route53 alias: `digests.<domain>` -> CloudFront.

### Phase 4 — Security headers and hardening
1. Add CloudFront response headers policy:
   - `Strict-Transport-Security`
   - `X-Content-Type-Options`
   - `Referrer-Policy`
2. Add WAF (from cost plan) before go-live.

---

## Route behavior checks
Validate these over HTTPS:
- `/`
- `/view/?id=<digest-id>`
- `/view/<digest-id>.md`
- `/search/`
- `/search/?q=test`

Also verify redirects:
- `http://...` -> `https://...`

---

## Acceptance criteria
- Public URL is HTTPS-only.
- HTTP redirects to HTTPS.
- Direct S3 public access is blocked.
- TLS cert is valid and auto-renewing via ACM.
- Site routes function unchanged behind CloudFront.
