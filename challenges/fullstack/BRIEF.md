# Take-Home Task: **Deal Pipeline & Document Vault**
React 18+ *or* Vue 3 (Frontend) | Node.js + Express (Backend)

## 🎯 Objective
Build a full-stack app that lets an M&A analyst:
1. Authenticate with a **passwordless email flow, implemented headlessly** — no hosted/drop-in auth widget
2. Search a catalogue of companies and add them to a personal watchlist
3. Move a company through a **deal pipeline** and attach supporting documents to each stage
4. Have the backend **verify and register** each document upload, returning a stable receipt

## 🔧 Requirements

### 🧩 Frontend (React 18+ or Vue 3 — your choice, TypeScript)
* Implement the email authentication flow **headlessly**: your own screens for `email → one-time code → session`. ⚠️ Do not drop in a pre-built auth widget ⚠️
* After authentication:
   * Show the signed-in analyst and a **watchlist** of companies
   * Provide a search over the company catalogue (name, sector)
   * Let the user add/remove companies from the watchlist
   * For a selected company, show a **pipeline** of stages (`Sourcing → NDA → Due diligence → Offer`) and let the user advance it
   * Let the user **attach a document** to the current stage and see the backend's verdict
* Show the result from the backend:
   * Whether the document was accepted, and its **checksum receipt**
   * Whether it was a new document or a re-upload of one already on file
* Display a **local activity history** of everything done in the session

**Note:** How you structure the app is up to you — pick React or Vue, whichever you write best; we judge the patterns, not the framework. The app complexity is high enough that good ones will shine through.

### 🌐 Backend (Node.js + Express – required)
Create a small REST API:

* `POST /auth/request-code` → `{ email }` — issues a one-time code (log it to the console; no mail provider needed)
* `POST /auth/verify-code` → `{ email, code }` — returns a session token
* `POST /auth/logout` — revokes the session server-side
* `GET  /companies?q=&sector=` — search the seeded catalogue
* `GET  /watchlist` · `POST /watchlist` · `DELETE /watchlist/:companyId`
* `PATCH /watchlist/:companyId/stage` → `{ stage }` — advance the pipeline, rejecting illegal transitions
* `POST /documents` → `{ companyId, stage, filename, contentBase64 }`

Every route but `/auth/*` requires a valid session token.

**The auth flow is where we look first.** Three things are expected, and they are enough:
* The one-time code **expires** (a few minutes) and is **single-use** — replaying it fails
* Verification attempts are **rate-limited**, so the 6-digit code cannot simply be brute-forced
* `logout` invalidates the session **server-side**, not merely client-side

If you go further (hashed codes, no account enumeration, refresh-token rotation), say so in the README — but a clean, well-tested version of the three above beats a half-finished version of everything.

The document endpoint must be **idempotent**: compute a SHA-256 checksum of the content and treat it as the identity of the document.

Return:
```json
{
  "accepted": true,
  "created": false,
  "checksum": "9f86d081...",
  "documentId": "doc_123",
  "companyId": "cmp_42",
  "stage": "due_diligence"
}
```
`created: true` on a genuine first upload (HTTP 201), `created: false` when the same content is already on file (HTTP 200) — a file re-sent under a different name must **not** produce a duplicate.

Seed the catalogue with ~30 companies from a fixture file (name, sector, city, revenue, headcount). No database required.

## Behavior & Constraints
* Session state can be in-memory (no DB required)
* Watchlist and activity history should persist across React component state or localStorage
* No third-party file-integrity or auth service — compute checksums, hash codes and validate sessions yourself (`node:crypto` or similar). A JWT library is fine; an authentication *provider* is not.
* Illegal pipeline transitions (e.g. `Sourcing → Offer`) must be rejected by the backend, not merely hidden in the UI

## ⏱️ Scope
Aim for **4–6 hours**. We would rather see a small, clean, tested app than a wide one that is half-wired — if you run short, cut features and say so in the README.

## 🚀 Submission Guidelines
* Submit a **PR to your GitHub repo and invite yassine.bouderbala@takeovers.ai on it for a review**
* Include:
   * Setup instructions for both frontend and backend in a README.md file
   * Notes on any trade-offs made or areas you'd improve
   * A test suite covering at least the one-time-code rules and the idempotent upload — all tests passing
* Bonus: Harden the auth further — hashed codes, refresh-token rotation, or a TOTP second factor
* Bonus: Per-stage document access — a file attached at `Due diligence` stays hidden while the company sits at `Sourcing`
* Bonus: Link to a deployed version (e.g. Vercel frontend, Render backend)

## ✅ Evaluation Focus
| Area | Evaluated On |
|------|-------------|
| **Frontend architecture** | Component design, state flow, composition, separation of concerns |
| **Auth flow** | Headless login UX, token lifecycle, session context, protected routes, and how the threats above are handled |
| **Node.js + Express** | REST API correctness, idempotency logic, state machine, modularity |
| **Code quality** | Readability, organization, error handling, TypeScript use |
| **User experience** | Clear flows, responsive feedback, intuitive UI |
| **Extensibility** | Evidence of scalable thought (e.g. room for roles, permissions, new stages, audit trails) |
| **Design** | Beautiful UX design skills are important to us. Make the app look and feel great |
