# PGx Risk Dashboard – Test Checklist (PHTS as Reference)

Use this checklist to verify tab behavior, code entry, and Risk Assessment display. **PHTS is the working reference**: compare behavior at https://jerome-dixon.io/uva/phts-risk-calculator/ when troubleshooting.

**PGx (VCU):**
- Custom domain: https://jerome-dixon.io/vcu/pgx-risk-calculator/
- S3 direct: https://jerome-dixon.io.s3.us-east-1.amazonaws.com/vcu/pgx-risk-calculator/index.html

---

## 1. Tab content visibility

**Reference (PHTS):** Every tab (Baseline Model, Extended Model, Model Comparison, Causal Analysis, Documentation) shows its content when clicked; only one panel is visible at a time.

| # | Test | Expected | Pass |
|---|------|----------|------|
| 1.1 | Open PGx dashboard (custom domain or S3 URL). | Risk Assessment tab content is visible (age, codes summary, Calculate button). | |
| 1.2 | Click **Drugs** (top-level tab). | Code-entry panel appears with Drugs sub-tab (search, matches, selected chips). | |
| 1.3 | Click **CPT Codes** (top-level tab). | Same code-entry panel with CPT Codes sub-panel visible. | |
| 1.4 | Click **ICD Codes** (top-level tab). | Same code-entry panel with ICD Codes sub-panel visible. | |
| 1.5 | Click **Causal Analysis** (lower-level tab). | Causal Analysis content (cohort/age band, Load button) visible. | |
| 1.6 | Click **DTW Trajectories**, **FP-Growth Patterns**, **BupaR Process Mining**, **PGx Patient Card**, **Documentation**. | Each tab shows its own content; no blank panel. | |
| 1.7 | Click **Risk Assessment** again. | Risk Assessment content visible again. | |

**If tabs other than Risk Assessment show no content:**  
- Confirm the deployed `index.html` sets `content.style.display` in `switchTab()` (JS) and runs the init block that sets `.tab-content` display on load.  
- Compare with PHTS: open DevTools → Elements, click a PHTS tab, confirm the corresponding content div has `display: block` and others `display: none`.

---

## 2. Drugs / ICD / CPT code entry

**Reference (PHTS):** Inputs are on the same tab as the calculator. PGx uses separate tabs for code entry; they must still be clearly available.

| # | Test | Expected | Pass |
|---|------|----------|------|
| 2.1 | Click **Drugs** tab. Enter age 35 on Risk Assessment first if needed, then open Drugs. | Search box and (after metadata load) drug list/chips UI visible. | |
| 2.2 | Search for a drug, click to add. | Drug appears as selected (chip or list). | |
| 2.3 | Click **ICD Codes** tab. | ICD search and selection UI visible; can add ICD codes. | |
| 2.4 | Click **CPT Codes** tab. | CPT search and selection UI visible; can add CPT codes. | |
| 2.5 | For age 65+ (Polypharmacy), open Drugs / ICD / CPT tabs. | Drugs always available; ICD/CPT may be hidden for polypharmacy per design. | |

**If code-entry tabs don’t show:** Fix tab visibility first (Section 1). Then confirm `#patient-codes-tab` and its sub-panels (Drugs / ICD / CPT) are shown when the Drugs, ICD Codes, or CPT Codes top-level tabs are clicked.

---

## 3. Selected drugs / ICD / CPT display on Risk Assessment

**Reference:** User should see what’s selected for the model in one place (e.g. PHTS shows form values on the same tab). PGx shows a summary plus read-only lists.

| # | Test | Expected | Pass |
|---|------|----------|------|
| 3.1 | On Risk Assessment, with no codes selected. | Summary: “No codes selected. Go to Drugs, CPT Codes, or ICD Codes…” and “Selected drugs for model”, “Selected ICD codes for model”, “Selected CPT codes for model” (read-only dropdowns/lists empty or minimal). | |
| 3.2 | Add 1–2 drugs on Drugs tab, return to Risk Assessment. | Summary shows “Selected: N drug(s)…” and **Selected drugs for model** list/dropdown shows those drugs. | |
| 3.3 | Add ICD and CPT codes, return to Risk Assessment. | **Selected ICD codes for model** and **Selected CPT codes for model** show the selected codes. | |
| 3.4 | Click **Edit codes**. | Switches to code-entry tab (Drugs sub-tab). | |

**If drug dropdown/list is missing:** Confirm `#selected-codes-display` and `#selected-drugs-readonly`, `#selected-icds-readonly`, `#selected-cpts-readonly` exist and are updated by `updateSelectedCodesDisplay()` when switching to Risk Assessment and when summary is updated.

---

## 4. Deployment and cache

| # | Test | Expected | Pass |
|---|------|----------|------|
| 4.1 | After code changes, run `aws s3 sync 9_risk_dashboard/frontend s3://...` and CloudFront invalidation for `/vcu/pgx-risk-calculator/*`. | New HTML/JS/CSS live after invalidation completes. | |
| 4.2 | Open dashboard with hard refresh (Ctrl+F5 / Cmd+Shift+R). | No stale tab or styling behavior. | |
| 4.3 | Prefer opening via **custom domain** (e.g. jerome-dixon.io/vcu/...) rather than S3 direct URL. | Same origin as other assets; fewer CORS/cache quirks. | |

---

## 5. Quick PHTS vs PGx comparison

| Behavior | PHTS | PGx |
|----------|------|-----|
| Tab switch | `onclick="switchTab('tab-id')"`; content div `id="{tab-id}-tab"`; visibility by class or style. | Same pattern; `switchTab(tabName, codeSubTab)`; JS sets `content.style.display` and `.active`. |
| Where code entry lives | N/A (clinical form on same tab). | Separate tabs: Drugs, CPT Codes, ICD Codes (one panel, sub-tabs). |
| Showing selected inputs | Form values on same tab. | Summary text + read-only “Selected drugs / ICD / CPT for model” on Risk Assessment. |

When in doubt, compare DOM and computed styles on PHTS (working) vs PGx (failing) for the same action (e.g. “click second tab”) and align PGx markup/JS accordingly.
