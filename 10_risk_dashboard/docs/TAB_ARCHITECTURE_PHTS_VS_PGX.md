# Tab Architecture: PHTS vs PGx Risk Dashboard

Comparison of the PHTS (UVA) and PGx (VCU) risk calculator dashboard tab structure and where inputs live.

**References:**
- **PHTS:** `C:\Projects\phts\graft-loss\cohort_analysis\calculator\risk_dashboard\phts_dashboard.html`  
  Live: https://jerome-dixon.io/uva/phts-risk-calculator/
- **PGx:** `10_risk_dashboard/frontend/index.html`  
  Live: https://jerome-dixon.io/vcu/pgx-risk-calculator/

---

## PHTS Tab Architecture

| Tab | Purpose | Where inputs live |
|-----|---------|-------------------|
| **Baseline Model** | Risk calculator (baseline features) | **Same tab:** cohort dropdown + form (eGFR, BUN, creatinine, LVAD, ECMO, etc.) + Calculate / Clear / Load Baseline |
| **Extended Model** | Risk calculator (extended features) | **Same tab:** same pattern – all inputs and Calculate on this tab |
| **Model Comparison** | Side‑by‑side Baseline vs Extended | Reads from both calculator tabs; no separate input tab |
| **Causal Analysis** | Causal factors, charts, sliders | Cohort selector + controls on this tab; optional load from calculator |
| **Documentation** | Link out (navigates away) | `window.location.href='phts_readme.html'` |

**Mechanism:** `onclick="switchTab('baseline-model')"` (etc.). Tab content id = `{tabName}-tab`. Visibility = `.tab-content` / `.tab-content.active` (display: none / block). Active tab button matched by **button text** (e.g. "Baseline Model").

**Important:** In PHTS, **all risk inputs are on the calculator tab itself**. There is no separate “input” or “codes” tab. User stays on Baseline (or Extended), fills the form, and clicks Calculate.

---

## PGx Tab Architecture

| Tab | Purpose | Where inputs live |
|-----|---------|-------------------|
| **Risk Assessment** | Age + risk calculation + results | **Age** and **summary of selected codes** on this tab; **“Edit codes”** opens the codes tab. No drug/ICD/CPT inputs here. |
| **Drugs, ICD & CPT Codes** | Select drugs, ICD, CPT for the model | **Separate tab** with sub‑tabs: **Drugs \| ICD Codes \| CPT Codes**. Search + click to add; chips for selected. Selections feed Risk Assessment. |
| **Causal Analysis** | Causal visualizations | Cohort/age band selectors and content on this tab |
| **DTW Trajectories** | DTW visualizations | Cohort/age band on this tab |
| **FP-Growth Patterns** | FP-Growth visualizations | Cohort/age band/item type on this tab |
| **BupaR Process Mining** | BupaR visualizations | Cohort/age band on this tab |
| **PGx Patient Card** | PGx card from SNP data | Inputs and Generate on this tab |
| **Documentation** | How to use the dashboard | **Same page:** inline content in tab (no link out) |

**Mechanism:** `data-tab="risk-assessment"` (etc.). Click handler finds content by `id="${targetTab}-tab"` and toggles `content.style.display` and `.active` on buttons. No inline onclick.

**Important:** In PGx, **code selection is on a dedicated tab**. Risk Assessment only shows age, a short summary (“Selected: N drugs, M ICDs, K CPTs”), and “Edit codes” that switches to the Drugs, ICD & CPT Codes tab.

---

## Main Differences

| Aspect | PHTS | PGx |
|--------|------|-----|
| **Inputs and calculator** | Inputs (clinical form) and Calculate are **on the same tab** (Baseline / Extended). | **Split:** code selection is on **“Drugs, ICD & CPT Codes”**; Risk Assessment has age + summary + “Edit codes” + Calculate. |
| **Number of top-level tabs** | 5 (Baseline, Extended, Comparison, Causal, Documentation). | 8 (Risk Assessment, Drugs/ICD/CPT, Causal, DTW, FP-Growth, BupaR, PGx Card, Documentation). |
| **Model/calculator structure** | Two calculator tabs (Baseline vs Extended) + Model Comparison tab. | One **Risk Assessment** tab; model chosen by **cohort** (Opioid ED or Polypharmacy) and age band. Single calculator; cohort selects which model. |
| **Sub-tabs** | None. | **Drugs, ICD & CPT Codes** has sub-tabs: Drugs \| ICD Codes \| CPT Codes. |
| **Tab switching** | Inline `onclick="switchTab('tab-name')"`. Active button by **text** (e.g. "Baseline Model"). | `data-tab` + `addEventListener`. Content by `id="${targetTab}-tab"`. |
| **Documentation** | Dedicated tab that links to `phts_readme.html` (navigates away). | **Documentation** tab shows content on the same page (inline). |
| **Input type** | Numeric and binary **clinical features** (eGFR, BUN, LVAD, etc.) in a **form**. | **Codes** (drugs, ICD, CPT) chosen from **search + multi-select/chips** driven by metadata. |

---

## Summary

- **PHTS:** One place per calculator – each risk tab (Baseline, Extended) contains both the inputs and the Calculate action. Simple “form on same tab” pattern.
- **PGx:** Code selection is separated into a **“Drugs, ICD & CPT Codes”** tab (with sub-tabs) and is **linked** to Risk Assessment via summary + “Edit codes.” Risk Assessment stays focused on age and running the model; code entry is centralized on the codes tab.

If you want PGx to behave more like PHTS (inputs on the same tab as the calculator), you could move the Drugs/ICD/CPT selection UI onto the Risk Assessment tab (e.g. expandable sections or the same sub-tabs inline) and optionally keep or remove the separate “Drugs, ICD & CPT Codes” tab.
