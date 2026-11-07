# Compliance, Security & HIPAA Architecture Specification
## Data Privacy, Regulatory Scope, Ephemeral In-Memory Execution & Enterprise AWS HIPAA/BAA Extension

**Platform:** Serverless Pharmacogenomic & Polypharmacy Risk Triage Platform (`pgx-risk-calculator`)  
**Live Production Interface:** [https://pgx.jerome-dixon.io/](https://pgx.jerome-dixon.io/)  
**Authors:** R. Jerome Dixon, MDA, MSc IT & Elvin T. Price, Pharm.D., Ph.D., FAHA  
**Academic Affiliation:** Virginia Commonwealth University (Integrative Life Sciences, School of Pharmacy / Department of Pharmacotherapy & Outcomes Science)  
**Related Grant:** VCU Wright Center for Clinical and Translational Research (CCTR) $25,000 Pilot Grant (*REDCap ID 3APJPFWR / 1a18dfe1eae3a835*)  
**Companion Manuscripts:** *Clinical and Translational Science* (Wiley, Accepted/Published: [10.1111/cts.70697](https://doi.org/10.1111/cts.70697), [10.1111/cts.70718](https://doi.org/10.1111/cts.70718), [10.1111/cts.70690](https://doi.org/10.1111/cts.70690))  
**Document Version:** 2.2 (Updated Post-Committee Review · November 2026)

---

## Executive Summary

This specification governs the data privacy, regulatory classification, and security architecture of the **Serverless Pharmacogenomic & Polypharmacy Risk Dashboard**. 

Healthcare machine learning systems often face an adoption bottleneck: monolithic hospital on-premises servers require extensive capital and maintenance, while public cloud architectures introduce severe data-at-rest cyber liability and complex Business Associate Agreement (BAA) encumbrances. To resolve this dilemma, our platform provides a **dual-architecture deployment model**:

```
+--------------------------------------------------------------------------------------------------+
|                                    DUAL DEPLOYMENT ARCHITECTURES                                 |
+-------------------------------------------------+------------------------------------------------+
|  MODE 1: NATIVE STATELESS TRIAGE (DEFAULT)      |  MODE 2: ENTERPRISE PERSISTENT PHI (OPT-IN)    |
+-------------------------------------------------+------------------------------------------------+
| • Primary Target: Safety-Net Clinics & MTM      | • Primary Target: Hospital Networks & EHRs     |
| • Architecture: "Compute-Respond-Discard"       | • Architecture: AWS HIPAA-Eligible Enclave     |
| • Persistent Storage: 0 PHI At Rest (RAM Only)  | • Persistent Storage: KMS-Encrypted RDS/S3     |
| • Operating Cost: < $5.00 / month               | • BAA Status: Formal Executed AWS BAA          |
| • BAA Requirement: None for maintainers         | • Compliance: 45 CFR Part 164 Subpart C        |
| • Execution Latency: 6 ms warm (< 100 ms SLA)   | • Telemetry: Immutable CloudTrail Audit Logs   |
+-------------------------------------------------+------------------------------------------------+
```

> [!IMPORTANT]
> **Core Architecture Statement:**  
> Our **native use case** is strictly **Stateless Zero-PHI Triage**: patient features are evaluated in ephemeral Lambda container RAM and instantly discarded upon response delivery, eliminating data-at-rest liability for resource-constrained safety-net pharmacies.  
> **However, if a user, healthcare provider, or health system wants or requires storing Protected Health Information (PHI)**—such as maintaining longitudinal patient registries, re-identifying cohorts, or integrating bi-directional EHR order workflows—**the platform seamlessly transitions to a fully compliant AWS HIPAA-Eligible Architecture governed by an executed Business Associate Addendum (BAA)**.

---

## 1. Regulatory Context, Scope Clarification & Committee Governance

### Scope Clarification: Patient vs. Healthcare Organization

Your first premise is **accurate**: HIPAA rules and BAA requirements apply exclusively when a regulated entity (a covered entity like a hospital/clinic, or another Business Associate) uses your tool to process patient data. When an individual patient directly inputs their own personal health data for their own use, HIPAA does not govern that relationship, and no BAA is required or legally valid.

```
+--------------------------------------------------------------------------------------------------+
|                                REGULATORY SCOPE DEMARCATION                                      |
+-------------------------------------------------+------------------------------------------------+
| 1. HEALTHCARE ORGANIZATION (COVERED ENTITY)     | 2. INDIVIDUAL PATIENT (DIRECT-TO-CONSUMER)     |
+-------------------------------------------------+------------------------------------------------+
| • Regulated by: HIPAA (HHS OCR), HITECH         | • Regulated by: FTC Act (§ 5), FTC HBNR        |
| • Data: ePHI (Protected Health Information)     | • Data: Consumer Health Data (Non-ePHI)        |
| • BAA Status: Mandatory if vendor hosts ePHI    | • BAA Status: Not applicable (legally invalid) |
| • Responsibility: Enterprise IT & Compliance    | • Responsibility: Individual End User          |
+-------------------------------------------------+------------------------------------------------+
```

### Committee Governance (Dr. Tamas Gal & Dr. Elvin Price)

During the doctoral defense review process, committee member **Dr. Tamas Gal** (Director of Research Informatics, VCU Massey Comprehensive Cancer Center) and Dissertation Chair **Dr. Elvin T. Price** highlighted the essential legal and regulatory nuance governing transient healthcare data:

> *"The claim that a 'compute, respond, discard' architecture permits use of HIPAA data without a Business Associate Agreement also needs institutional privacy/legal review. Lack of persistent storage does not by itself establish that a cloud service is outside HIPAA business-associate requirements under 45 CFR § 160.103."*

### Resolution via VCU Wright Center CCTR $25,000 Pilot Grant

Rather than making an unvetted categorical claim that ephemeral in-memory processing automatically exempts commercial cloud transit from HIPAA obligations, we directly operationalized this inquiry into our funded **VCU Wright Center CCTR $25,000 Pilot Grant** (*Serverless Pharmacogenomic and Polypharmacy Risk Triage: Prospective Validation and Pharmacist Usability Testing*, REDCap `3APJPFWR` / `1a18dfe1eae3a835`):

1. **Aim 2 Governance Funding ($4,500):** Specifically funds an institutional legal and privacy review in partnership with **VCU Health Compliance**, **VCU Massey Cancer Center Research Informatics Governance**, and **VCU University Counsel**.
2. **Statutory Audit Objectives:**
   - Formalize the exact regulatory boundaries of ephemeral AWS Lambda `/tmp` container RAM under HHS Office for Civil Rights (OCR) enforcement precedents.
   - Establish standard institutional Business Associate Agreements (BAAs) and data transmission contracts before routing live hospital patient encounters.
   - Evaluate containerization requirements for deploying private, on-premise VPC enclaves within institutional health-system firewalls.

---

## 2. Mode 1: Native Stateless Triage (Default / Zero-PHI)

Mode 1 is the default open-source and clinical prototype footprint evaluated in Chapter 5 (*Clinical and Translational Science*, Accepted; DOI: [10.1111/cts.70697](https://doi.org/10.1111/cts.70697)).

```
                       MODE 1: NATIVE STATELESS PIPELINE
                       
   [ Clinician / Bedside Browser ]
                 │
                 ▼ HTTPS (TLS 1.3 / Strict CORS)
      ┌─────────────────────┐
      │  Amazon CloudFront  │  <── Global CDN Edge & TLS Termination
      └──────────┬──────────┘
                 │
                 ▼ REST API Request (JSON Payload)
      ┌─────────────────────┐
      │ Amazon API Gateway  │  <── Payload Throttling, WAF & Schema Validation
      └──────────┬──────────┘
                 │
                 ▼ Ephemeral Event Trigger
      ┌────────────────────────────────────────────────────────┐
      │              AWS Lambda Container MicroVM              │
      │                                                        │
      │  1. Ingest Payload into Volatile Heap (RAM)            │
      │  2. Two-Key Partition Routing (Age Band × Med Count)   │
      │  3. Execute 3 Base Learners (CatBoost + 2x XGBoost)    │
      │  4. Compute Consensus Attribution & Top-5 SHAP         │
      │  5. Query CPIC Cache (573 Deterministic Rules)         │
      │  6. Return JSON Response to Client                     │
      │  7. IMMEDIATE MEMORY FLUSH (Execution Ends)            │
      │                                                        │
      │  [ ZERO DISK WRITES • ZERO LOGICAL STORAGE • NO DB ]   │
      └────────────────────────────────────────────────────────┘
```

### Technical & Privacy Specifications

* **Stateless Compute-Respond-Discard:** Patient feature vectors (age, sex, and binary medication/diagnosis codes) exist only within volatile system memory for the duration of request processing (**mean warm latency: `6 ms`, SD `1 ms`**; PGx card generation: `60 ms`).
* **Zero Disk Persistence:** No temporary files, SQLite caches, or diagnostic payloads are written to the container's `/tmp` filesystem. The ephemeral microVM environment freezes and destroys memory buffers immediately upon execution termination.
* **No Database at Rest:** The architecture maintains no Amazon RDS, DynamoDB, or external database connections. Model artifacts (`.cbm`, `.json`) and CPIC guidelines (`cpic_cache.json`) are pre-baked as read-only image layers or mounted as read-only S3 assets.
* **Input Sparsity Resilience ("Imputation of Normality"):** Bedside clinicians enter sparse medication lists (`2–5` active drugs) without full APCD claims histories. The algorithm initializes unentered covariates to cohort training medians ($\mathbf{X}_{\text{full}} = \text{median}_{\text{train}}[\text{age}_{\text{band}}]$), preserving prediction fidelity ($|\Delta\hat{p}| < 0.06$) up to **70% input missingness** and maintaining **87% risk band stability**.
* **Zero Telemetry Guarantee:** Contains no phone-home analytics, advertising trackers, or error-reporting telemetry. All outbound network traffic is strictly isolated.
* **Operating Cost:** Operates for less than **$5.00/month** under active evaluation workloads, fulfilling the Chapter 2 economic thesis of scalable, low-cost clinical AI for rural and community providers.

---

## 3. Mode 2: Enterprise Persistent PHI Architecture (Opt-In / HIPAA-Eligible)

When a health system, accountable care organization (ACO), or clinical research study requires storing Protected Health Information (PHI/ePHI)—for longitudinal patient tracking, cohort re-identification, automated EHR order-entry, or clinical trial follow-up—the platform transitions to a fully compliant **AWS HIPAA-Eligible Architecture**.

### The Legal BAA Execution Chain

When a healthcare organization uses your hosted service, the agreement chain involves two separate contracts rather than a single direct link between the hospital and your AWS account:

```
[ Healthcare Organization ] ── ( BAA #1 ) ──► [ Your Business / Legal Entity ] ── ( BAA #2 ) ──► [ AWS ]
```

1. **Covered Entity to Vendor (You):** You (or your legal entity/company) sign a BAA directly with the healthcare organization. The BAA is executed with your legal entity, not your AWS account number.
2. **Vendor (You) to Cloud Provider (AWS):** You sign a Business Associate Addendum with AWS through **AWS Artifact** in your AWS Management Console. This legally covers AWS as your subcontractor for the underlying infrastructure processing the ePHI.

```
                   MODE 2: ENTERPRISE HIPAA-COMPLIANT ARCHITECTURE
                   
   [ Enterprise EHR (Epic / Cerner) ]  <── SMART-on-FHIR / OAuth 2.0
                 │
                 ▼ AWS Direct Connect / VPN / PrivateLink
      ┌───────────────────────────────────────────────────────────────┐
      │                   AWS Enterprise VPC Enclave                  │
      │                                                               │
      │  ┌────────────────────────┐       ┌────────────────────────┐  │
      │  │   Public Subnet        │       │   Private Subnet       │  │
      │  │                        │       │                        │  │
      │  │  AWS WAF + API Gateway │ ────► │  AWS Lambda MicroVMs   │  │
      │  │  (TLS 1.3 Termination) │       │  (VPC-Attached, No IGW)│  │
      │  └────────────────────────┘       └───────────┬────────────┘  │
      │                                               │               │
      │               ┌───────────────────────────────┴────────┐      │
      │               ▼                                        ▼      │
      │  ┌────────────────────────┐               ┌────────────────┐  │
      │  │   Amazon Aurora PG     │               │   Amazon S3    │  │
      │  │   (KMS Encrypted Rest) │               │   Audit Bucket │  │
      │  │   • Patient Identifiers│               │   • CloudTrail │  │
      │  │   • Longitudinal EMR   │               │   • ObjectLock │  │
      │  └────────────────────────┘               └────────────────┘  │
      │                                                               │
      │  [ AWS Business Associate Addendum (BAA) Executed ]           │
      │  [ AWS KMS Customer-Managed Keys (FIPS 140-3 HSM) ]           │
      │  [ AWS GuardDuty + CloudWatch Immutable Log Stream ]          │
      └───────────────────────────────────────────────────────────────┘
```

---

### AWS Technical Reality: "HIPAA-Eligible" vs. "HIPAA-Compliant"

Switching to AWS Lambda and S3 is necessary, but AWS operates on a **Shared Responsibility Model**. AWS provides *HIPAA-eligible* services, but turning them into a *HIPAA-compliant* application requires specific configurations on your end. There are no separate "HIPAA Lambdas"—the same standard services are used under specific security controls.

#### Required Technical Configurations for Lambda & S3

| Component | AWS Requirement | Mandatory Configuration |
| :--- | :--- | :--- |
| **AWS Account BAA** | AWS Artifact | Log into AWS Artifact and accept the AWS Business Associate Addendum before routing ePHI. |
| **S3 Buckets** | Encryption at Rest | Enforce AWS KMS encryption (`SSE-KMS`), disable public access completely, and enforce TLS (`aws:SecureTransport`) in bucket policies. |
| **AWS Lambda** | Environment & Memory Security | Encrypt environment variables using KMS. Ensure temporary storage (`/tmp`) does not persist ePHI across invocations. |
| **Logging (CloudWatch)** | Prevent ePHI Leaks | **Critical:** Ensure print/log statements inside your Lambda functions do not output raw request/response payloads containing ePHI into AWS CloudWatch logs. |
| **Audit Logging** | AWS CloudTrail | Enable CloudTrail to capture management and data API events for compliance auditing under 45 CFR § 164.312(b). |

---

### Important Nuance for AI & External API Workflows

If your Lambda function calls third-party APIs (such as external LLMs or data enrichment services):

* **AWS-Native Models:** Using **Amazon Bedrock** or models hosted on **AWS SageMaker** within your AWS account falls under your existing AWS BAA.
* **External APIs (e.g., OpenAI, Anthropic):** If your Lambda makes outbound HTTPS calls to an external API service outside AWS, **that external vendor must also sign a BAA with you**, regardless of whether their processing is ephemeral. Routing ePHI to a third-party API without a BAA violates HIPAA compliance even if your AWS environment is fully secured.

---

### HIPAA Security Rule Compliance Matrix (45 CFR Part 164 Subpart C)

| HIPAA Security Standard | Regulatory Citation | Enterprise Implementation Specification |
| :--- | :--- | :--- |
| **Business Associate Agreement** | 45 CFR § 164.502(e)<br>45 CFR § 164.504(e) | Formal execution of the **AWS Business Associate Addendum (BAA)** covering all provisioned services (API Gateway, Lambda, RDS, S3, KMS, CloudTrail). |
| **Access Control (RBAC)** | 45 CFR § 164.312(a)(1) | Strict IAM Role-Based Access Control (RBAC) enforcing least privilege. Integration with hospital Active Directory / Okta via SAML 2.0 / OAuth 2.0. |
| **Unique User Identification** | 45 CFR § 164.312(a)(2)(i) | Every clinician interaction authenticated via SMART-on-FHIR JWT access tokens with unique NPI/user claim validation. |
| **Emergency Access ("Break-Glass")**| 45 CFR § 164.312(a)(2)(ii) | Pre-configured emergency break-glass IAM roles triggering immediate high-priority alerts to hospital Information Security Officers (ISOs). |
| **Automatic Logoff** | 45 CFR § 164.312(a)(2)(iii)| Client application sessions enforce automated token expiration and DOM clearing after 15 minutes of inactivity. |
| **Encryption at Rest** | 45 CFR § 164.312(a)(2)(iv)| **AES-256 GCM** encryption enforced across Amazon Aurora PostgreSQL, DynamoDB tables, and S3 buckets using **AWS KMS Customer Managed Keys (CMKs)** with automated annual key rotation. |
| **Audit Controls** | 45 CFR § 164.312(b) | **AWS CloudTrail** captures all management and data events. Log files are delivered to a dedicated, write-once-read-many (WORM) S3 bucket protected by **S3 Object Lock** in Compliance Mode. |
| **Data Integrity Verification** | 45 CFR § 164.312(c)(1) | SHA-256 cryptographic checksums generated for all stored risk evaluations and genomic diplotype records; automated SSL/TLS certificate verification. |
| **Transmission Security** | 45 CFR § 164.312(e)(1) | All communication in transit enforced over **TLS 1.3** using modern cryptographic cipher suites (`ECDHE-RSA-AES128-GCM-SHA256` or better). Unencrypted HTTP strictly blocked. |
| **Network Isolation** | 45 CFR § 164.312(e)(2)(i) | Lambda execution environments deployed inside private VPC subnets with no internet gateway. Communication with health system EHRs routed over **AWS Direct Connect**, **IPsec VPN**, or **AWS PrivateLink**. |
| **Security Incident Monitoring** | 45 CFR § 164.308(a)(1)(ii)(D) | **Amazon GuardDuty** continuous threat detection paired with **AWS Security Hub** and automated CloudWatch anomaly alarms. |

---

## 4. Direct Consumer & Patient Use (Direct-to-Consumer Genotypes)

When individual patients access the platform directly (e.g., uploading consumer `23andMe` or `AncestryDNA` raw genetic files in Tab 5), processing falls outside HIPAA covered-entity jurisdiction and is governed by federal consumer protection statutes and state health privacy laws:

```
+--------------------------------------------------------------------------------------------------+
|                            CONSUMER DATA GOVERNANCE FRAMEWORK                                    |
+------------------------------------+-------------------------------------------------------------+
| STATUTE / REGULATOR                | COMPLIANCE REQUIREMENT & IMPLEMENTATION                     |
+------------------------------------+-------------------------------------------------------------+
| FTC Act (Section 5)                | Prohibits deceptive or unfair privacy practices. Platform   |
| 15 U.S.C. § 45                     | adheres strictly to stated Zero-Persistence data promises.  |
+------------------------------------+-------------------------------------------------------------+
| FTC Health Breach Notification     | Enforces mandatory federal notification in the event of     |
| Rule (HBNR) — 16 CFR Part 318      | unauthorized acquisition of unsecured consumer health data. |
+------------------------------------+-------------------------------------------------------------+
| Washington My Health My Data Act   | Requires affirmative, opt-in consent prior to collecting or |
| (MHMDA) — RCW 19.373               | processing consumer health metrics; strictly bars selling.  |
+------------------------------------+-------------------------------------------------------------+
| California Consumer Privacy Act    | Grants patients the right to know, delete, and opt out. In  |
| (CCPA / CPRA) — Cal. Civ. Code     | Mode 1, data are instantly discarded, ensuring zero storage.|
+------------------------------------+-------------------------------------------------------------+
```

### Client-Side Genetic File Handling

* **In-Browser VCF/Text Parsing:** Consumer genotype files (`.txt`, `.vcf`) can be parsed locally in client-side JavaScript/WebAssembly. Only the specific variant identifiers matching the **`573` CPIC gene-drug guidelines** (e.g., `CYP2D6 *4`, `OPRM1 rs1799971`) are extracted; the remaining raw genomic sequence is never transmitted over the network.
* **No Genetic Data Monetization:** The platform does not sell, license, share, or aggregate patient genetic information under any circumstances.

---

## 5. FDA SaMD & Cures Act § 3060(a) Regulatory Classification

To ensure regulatory certainty for deploying healthcare institutions, the platform’s decision-support mechanics are engineered to align with the **21st Century Cures Act Section 3060(a)** and **FDA Software as a Medical Device (SaMD / 21 CFR Part 820)** guidelines.

### Transparent Non-Device Clinical Decision Support (CDS) Exemption

Under Section 520(o)(1)(E) of the FD&C Act (enacted by the Cures Act § 3060(a)), software is **exempt from FDA medical device regulation** if it satisfies all four statutory criteria:

1. **No Hardware/Signal Processing:** The software does not acquire, process, or analyze medical images, signals from monitoring hardware, or IVD laboratory waveforms.
2. **Display & Analysis of Medical Information:** The software displays and analyzes clinical information (diagnoses, medications, age, and genotypes) entered by clinicians or retrieved from claims/EHR records.
3. **Supports Professional Decision-Making:** The software is designed exclusively to support licensed healthcare professionals (clinical pharmacists, physicians) who independently evaluate the recommendations.
4. **Independent Clinician Review (Explainability Standard):** Crucially, the software enables the clinician to *independently review the basis for the recommendations* without relying on an opaque "black-box" model.
   - **Symbolic Formal Feature Attribution (FFA):** Outputs exact Boolean Horn-clause rules (`5-Set CNF Union`) detailing why an alert triggered.
   - **Counterfactual Intervention Rate ($\text{IR} > 0.10$):** Displays the exact model-perturbation delta when adjusting medications.
   - **Peer-Reviewed CPIC Citations:** Every pharmacogenomic recommendation links directly to published **CPIC Level A/B guidelines** and PubMed evidence.

### FDA SaMD Readiness & Predetermined Change Control Plan (PCCP)

If an enterprise deployer chooses to integrate automated order-blocking or closed-loop prescription modification, the platform transitions to **FDA SaMD Class II (`21 CFR 820.30` Design Controls)**:

* **Deterministic Verification:** Supported by a comprehensive automated test suite verifying **`573 / 573` (`100%`) CPIC guideline pairs** against standard reference diplotypes.
* **PCCP Model Drift Governance:** Enforces continuous CI/CD evaluation on GitHub Actions / Amazon ECR. An automated retrain-and-review workflow is triggered whenever the quarterly top-10 `|SHAP|` Spearman rank correlation drops by **`> 15%`** ($\rho < 0.85$), ensuring model calibration across evolving prescribing patterns.

---

## 6. Implementation & Hardening Checklist

Healthcare IT administrators and cloud architects deploying this repository should execute the following hardening checklist:

```
[ ] 1. Deployment Model Selection:
    [ ] Mode 1: Native Stateless Triage (Default / Zero-PHI / Scale-to-Zero)
    [ ] Mode 2: Enterprise Persistent PHI Enclave (Opt-In / HIPAA-Eligible)

[ ] 2. If Mode 1 (Native Stateless):
    [ ] Verify AWS Lambda environment has zero persistent disk mounts.
    [ ] Ensure CloudFront and API Gateway enforce HTTPS TLS 1.3.
    [ ] Confirm CORS origins are strictly restricted to authorized domains.
    [ ] Validate that API payload logging in CloudWatch does NOT log request body text.

[ ] 3. If Mode 2 (Enterprise HIPAA Enclave):
    [ ] Execute AWS Business Associate Addendum (BAA) in AWS Artifact.
    [ ] Deploy AWS CloudFormation / Terraform templates in dedicated AWS VPC private subnets.
    [ ] Provision AWS KMS Customer Managed Keys (CMKs) for database and storage encryption.
    [ ] Enable AWS CloudTrail with log file validation and send logs to S3 Object Lock bucket.
    [ ] Verify CloudWatch log scrubbers prevent printing ePHI request/response payloads.
    [ ] Configure SMART-on-FHIR OAuth 2.0 reverse proxy for EHR integration.
    [ ] Connect Amazon GuardDuty and configure AWS Security Hub alerting.
    [ ] If using AI/LLM features: Ensure models are AWS-native (Bedrock/SageMaker) or vendor signs BAA.

[ ] 4. Clinical Governance & Institutional Review:
    [ ] Complete institutional Risk Analysis under 45 CFR § 164.308(a)(1)(ii)(A).
    [ ] Synchronize with institutional privacy officer / IRB protocol guidelines.
```

---

## 7. Legal & Medical Disclaimers

### Legal Notice
This open-source software is provided "as is", without warranty of any kind, express or implied. Software authors, contributors, and Virginia Commonwealth University are not Business Associates of deploying entities merely by virtue of code distribution. Organizations deploying this software remain solely responsible for conducting their own Risk Analysis under **45 CFR § 164.308(a)(1)(ii)(A)**, verifying third-party cloud agreements, and ensuring institutional compliance.

### Medical & Clinical Disclaimer
This software is an analytical clinical decision support prototype designed strictly for research, educational, and professional clinical triage purposes. It does not provide automated diagnoses, clinical directives, or treatment plans. All recommendations, consensus attributions, and pharmacogenomic alerts must be independently evaluated and verified by a licensed clinical pharmacist, physician, or qualified healthcare professional before taking any clinical action.

---

*Specification maintained by the VCU Pharmacotherapy & Outcomes Science / Integrative Life Sciences Clinical ML Team.*  
*Correspondence: R. Jerome Dixon (`dixonrj@vcu.edu`) & Dr. Elvin T. Price (`etprice@vcu.edu`).*
