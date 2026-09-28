/* PGx Card governed workflow (Final PGx Dashboard Implementation Plan).
   CPIC actions require phenotype. Triplet alerts stay separate. APCD generics only. */
(function (global) {
  const ACTION_META = {
    AVOID_OR_USE_ALTERNATIVE: { label: "Avoid / alternative needed", rank: 0, cls: "pgx-act-avoid" },
    DOSE_REDUCTION_OR_TITRATION: { label: "Dose adjustment needed", rank: 1, cls: "pgx-act-dose" },
    DOSE_INCREASE_OR_ALTERNATIVE: { label: "Reduced response possible", rank: 2, cls: "pgx-act-dose" },
    ENHANCED_MONITORING: { label: "Monitoring recommended", rank: 3, cls: "pgx-act-monitor" },
    STANDARD_PRESCRIBING: { label: "No PGx action identified", rank: 4, cls: "pgx-act-std" },
    NO_CPIC_RECOMMENDATION: { label: "No applicable CPIC action", rank: 5, cls: "pgx-act-none" },
    INSUFFICIENT_GENOTYPE_RESOLUTION: { label: "Insufficient genetic resolution", rank: 6, cls: "pgx-act-indet" }
  };

  // Offline last-resort only. Happy path uses Lambda gene_calls + phenotypeTableVersion.
  const PHENOTYPE_TABLE = {
    CYP2C19: {
      "*2/*2": "Poor metabolizer", "*2/*3": "Poor metabolizer", "*3/*3": "Poor metabolizer",
      "*1/*2": "Intermediate metabolizer", "*1/*3": "Intermediate metabolizer",
      "*1/*1": "Normal metabolizer", "*1/*17": "Rapid metabolizer", "*17/*17": "Ultrarapid metabolizer",
      "*2/*17": "Intermediate metabolizer"
    },
    CYP2C9: {
      "*2/*2": "Poor metabolizer", "*3/*3": "Poor metabolizer", "*2/*3": "Poor metabolizer",
      "*1/*2": "Intermediate metabolizer", "*1/*3": "Intermediate metabolizer", "*1/*1": "Normal metabolizer"
    },
    CYP2D6: {
      "*4/*4": "Poor metabolizer", "*3/*4": "Poor metabolizer", "*4/*6": "Poor metabolizer",
      "*1/*4": "Intermediate metabolizer", "*1/*10": "Intermediate metabolizer", "*1/*41": "Intermediate metabolizer",
      "*1/*1": "Normal metabolizer", "*1/*2": "Normal metabolizer", "*2/*2": "Normal metabolizer"
    },
    SLCO1B1: { "*5/*5": "Poor function", "*1/*5": "Decreased function", "*1B/*5": "Decreased function", "*1/*1": "Normal function", "*1B/*1B": "Normal function" },
    CYP3A5: { "*3/*3": "Poor metabolizer", "*1/*3": "Intermediate metabolizer", "*1/*1": "Normal metabolizer" },
    CYP3A4: { "*22/*22": "Decreased function", "*1/*22": "Decreased function", "*1/*1": "Normal metabolizer", "*1B/*1B": "Normal metabolizer" },
    TPMT: { "*3A/*3A": "Poor metabolizer", "*3C/*3C": "Poor metabolizer", "*2/*2": "Poor metabolizer", "*1/*3C": "Intermediate metabolizer", "*1/*3B": "Intermediate metabolizer", "*1/*2": "Intermediate metabolizer", "*1/*1": "Normal metabolizer" },
    DPYD: { "*2A/*2A": "Poor metabolizer", "*1/*2A": "Intermediate metabolizer", "*1/*13": "Intermediate metabolizer", "*1/*1": "Normal metabolizer" },
    VKORC1: { "*2/*2": "High warfarin sensitivity", "*1/*2": "Increased warfarin sensitivity", "*1/*1": "Normal sensitivity" },
    CYP4F2: { "*3/*3": "Decreased function", "*1/*3": "Decreased function", "*1/*1": "Normal metabolizer" }
  };

  const AVOID_PAIRS = {
    CYP2C19: { "Poor metabolizer": ["clopidogrel", "citalopram"] },
    CYP2D6: { "Poor metabolizer": ["codeine", "tramadol"] },
    DPYD: { "Poor metabolizer": ["fluorouracil", "capecitabine"], "Intermediate metabolizer": ["fluorouracil", "capecitabine"] }
  };
  const DOSE_PAIRS = {
    CYP2C9: ["warfarin", "phenytoin", "celecoxib"],
    CYP2C19: ["voriconazole", "sertraline", "escitalopram"],
    CYP2D6: ["metoprolol", "atomoxetine", "ondansetron"],
    SLCO1B1: ["simvastatin", "atorvastatin", "rosuvastatin"],
    TPMT: ["azathioprine", "mercaptopurine", "thioguanine"],
    VKORC1: ["warfarin"],
    CYP3A5: ["tacrolimus"]
  };
  const CNS_OPIOIDS = ["oxycodone", "hydrocodone", "oxymorphone", "hydromorphone", "morphine", "fentanyl", "tramadol", "codeine", "buprenorphine", "methadone"];
  const CNS_GABAS = ["gabapentin", "pregabalin"];
  const CNS_BENZOS = ["alprazolam", "clonazepam", "diazepam", "lorazepam", "temazepam", "midazolam"];

  const state = {
    versions: {
      apcdVocabularyVersion: "apcd-generic-v1",
      cpicKnowledgeVersion: "cpic-gene-drug-pairs-dashboard",
      phenotypeTableVersion: "pgx-phenotype-v1",
      crosswalkVersion: "apcd-cpic-crosswalk-v1",
      tripletModelVersion: "regimen-three-way-v1",
      thresholdVersion: "triplet-threshold-v1",
      exportRendererVersion: "pgx-card-export-v3"
    },
    combinations: {},
    salts: {},
    aliases: {},
    vocab: [],
    selections: [],
    drugScope: "ACTIVE",
    viewMode: "clinician",
    actionableOnly: false,
    lastPayload: null,
    lastCalls: [],
    lastRecs: [],
    lastAlerts: []
  };

  const ACTIONABLE_CATEGORIES = {
    AVOID_OR_USE_ALTERNATIVE: true,
    DOSE_REDUCTION_OR_TITRATION: true,
    DOSE_INCREASE_OR_ALTERNATIVE: true,
    ENHANCED_MONITORING: true
  };
  const SALT_SUFFIXES = [
    "hydrochloride", "hcl", "sulfate", "sulphate", "bitartrate", "acetate", "sodium",
    "potassium", "calcium", "phosphate", "mesylate", "maleate", "tartrate", "citrate",
    "succinate", "furoate", "hydrobromide", "nitrate", "carbonate", "tosylate",
    "besylate", "pamoate", "fumarate"
  ];

  function slug(name) {
    return String(name || "").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  }
  function norm(name) {
    return String(name || "").trim().toLowerCase().replace(/\s+/g, " ");
  }
  function displayName(name) {
    return String(name || "")
      .replace(/^item_drug_/i, "")
      .replace(/^drug\s+/i, "")
      .replace(/_/g, " ")
      .trim();
  }

  function seedVocabExtras() {
    const extras = [];
    Object.values(AVOID_PAIRS).forEach((byPh) => {
      Object.values(byPh).forEach((arr) => extras.push.apply(extras, arr));
    });
    Object.values(DOSE_PAIRS).forEach((arr) => extras.push.apply(extras, arr));
    extras.push.apply(extras, CNS_OPIOIDS.concat(CNS_GABAS, CNS_BENZOS));
    Object.keys(state.combinations).forEach((k) => extras.push(k));
    Object.values(state.combinations).forEach((arr) => extras.push.apply(extras, arr));
    Object.keys(state.aliases || {}).forEach((k) => extras.push(k));
    Object.keys(state.salts || {}).forEach((k) => extras.push(k));
    extras.forEach((n) => {
      const d = displayName(n);
      if (d && !state.vocab.some((v) => norm(v) === norm(d))) state.vocab.push(d);
    });
    state.vocab.sort();
  }

  function diplotypeFromAlleles(alleles) {
    const stars = (alleles || []).map((a) => {
      const s = String(a || "").trim();
      if (!s || s === "0") return null;
      return s.startsWith("*") ? s : (s.match(/^rs/i) ? s : "*" + s.replace(/^\*/, ""));
    }).filter(Boolean);
    if (!stars.length) return null;
    if (stars.length === 1) return stars[0] + "/" + stars[0];
    return stars.slice(0, 2).sort().join("/");
  }

  function translatePhenotype(gene, alleles) {
    const g = String(gene || "").toUpperCase();
    const dip = diplotypeFromAlleles(alleles);
    const table = PHENOTYPE_TABLE[g] || {};
    if (!dip) {
      return { diplotype: null, phenotype: null, phenotypeConfidence: "INDETERMINATE", limitations: ["No allele call"] };
    }
    let ph = table[dip];
    if (!ph) {
      const parts = dip.split("/");
      const swapped = parts.slice().reverse().join("/");
      ph = table[swapped];
    }
    if (!ph) {
      return {
        diplotype: dip,
        phenotype: null,
        phenotypeConfidence: "INDETERMINATE",
        limitations: ["Allele pair not in phenotype table; treated as indeterminate, not normal"]
      };
    }
    return { diplotype: dip, phenotype: ph, phenotypeConfidence: "MODERATE", limitations: [] };
  }

  function classifyAction(gene, phenotype, drugName) {
    const g = String(gene || "").toUpperCase();
    const d = norm(displayName(drugName));
    if (!phenotype) return "INSUFFICIENT_GENOTYPE_RESOLUTION";
    const avoid = ((AVOID_PAIRS[g] || {})[phenotype] || []).map(norm);
    if (avoid.includes(d)) return "AVOID_OR_USE_ALTERNATIVE";
    const abnormal = /poor|intermediate|decreased|high warfarin|increased warfarin|rapid|ultrarapid/i.test(phenotype);
    if (abnormal && (DOSE_PAIRS[g] || []).map(norm).includes(d)) {
      if (/rapid|ultra/i.test(phenotype)) return "DOSE_INCREASE_OR_ALTERNATIVE";
      return "DOSE_REDUCTION_OR_TITRATION";
    }
    if (abnormal) return "ENHANCED_MONITORING";
    return "STANDARD_PRESCRIBING";
  }

  function buildGeneCalls(variants) {
    return (variants || []).map((v) => {
      const gene = String(v.gene || "").toUpperCase();
      const alleles = v.variants || v.alleleCalls || [];
      const tr = translatePhenotype(gene, alleles);
      return {
        gene,
        sourceVariants: alleles,
        alleleCalls: alleles,
        diplotype: tr.diplotype,
        phenotype: tr.phenotype,
        phenotypeConfidence: tr.phenotypeConfidence,
        limitations: tr.limitations,
        phenotypeTableVersion: state.versions.phenotypeTableVersion
      };
    }).filter((c) => c.gene);
  }

  function buildRecommendations(geneCalls, cpicDrugs, selectedNames) {
    const recs = [];
    const selected = new Set((selectedNames || []).map(norm));
    (cpicDrugs || []).forEach((drug) => {
      const gene = String(drug.gene || "").toUpperCase();
      const call = geneCalls.find((c) => c.gene === gene);
      if (!call) return;
      const name = displayName(drug.drug || drug.formattedGenericName);
      const action = call.phenotypeConfidence === "INDETERMINATE"
        ? "INSUFFICIENT_GENOTYPE_RESOLUTION"
        : classifyAction(gene, call.phenotype, name);
      recs.push({
        patientGeneCallId: gene,
        apcdDrugId: slug(name),
        formattedGenericName: name,
        gene,
        diplotype: call.diplotype,
        phenotype: call.phenotype,
        cpicGuidelineId: drug.guideline_url || drug.guideline || "",
        cpicGuidelineVersion: state.versions.cpicKnowledgeVersion,
        actionCategory: action,
        recommendationText: ACTION_META[action].label,
        cpicMappingStatus: "VERIFIED",
        sourceUrl: drug.guideline_url || drug.guideline || "",
        cpicLevel: drug.cpic_level || "",
        fdaLabel: drug.fda_label || drug.pgx_on_fda_label || "",
        evidenceType: action === "INSUFFICIENT_GENOTYPE_RESOLUTION" ? "Indeterminate genetic interpretation" : "CPIC-guideline action",
        inRegimen: selected.size ? [...selected].some((n) => nameMatches(name, n)) : false
      });
    });
    recs.sort((a, b) => {
      const ra = ACTION_META[a.actionCategory].rank - ACTION_META[b.actionCategory].rank;
      if (ra !== 0) return ra;
      if (a.inRegimen !== b.inRegimen) return a.inRegimen ? -1 : 1;
      return a.formattedGenericName.localeCompare(b.formattedGenericName);
    });
    return recs;
  }

  function comboKeyTokens(name) {
    return norm(displayName(name)).replace(/\s*\/\s*/g, " ").split(/\s+/).filter(Boolean).sort().join(" ");
  }

  function stripSaltName(name) {
    const raw = norm(displayName(name));
    if (state.salts[raw]) return state.salts[raw];
    const toks = raw.split(/\s+/);
    if (toks.length > 1 && SALT_SUFFIXES.indexOf(toks[toks.length - 1]) !== -1) {
      return toks.slice(0, -1).join(" ");
    }
    return displayName(name);
  }

  function lookupComboParts(name) {
    const raw = displayName(name);
    const key = norm(raw);
    const alias = state.aliases[key] || state.aliases[raw];
    if (alias && state.combinations[alias]) return state.combinations[alias];
    if (alias && state.combinations[norm(alias)]) return state.combinations[norm(alias)];
    if (state.combinations[key]) return state.combinations[key];
    if (state.combinations[raw]) return state.combinations[raw];
    const stripped = stripSaltName(raw);
    const strippedKey = norm(stripped);
    if (state.combinations[strippedKey]) return state.combinations[strippedKey];
    const want = comboKeyTokens(stripped);
    const keys = Object.keys(state.combinations);
    for (let i = 0; i < keys.length; i++) {
      if (comboKeyTokens(keys[i]) === want) return state.combinations[keys[i]];
    }
    return null;
  }

  function nameMatches(recName, selectedName) {
    const rec = norm(displayName(recName));
    const sel = norm(displayName(selectedName));
    if (!rec || !sel) return false;
    if (rec === sel) return true;
    const recTok = rec.split(/[^a-z0-9]+/).filter(Boolean);
    const selTok = sel.split(/[^a-z0-9]+/).filter(Boolean);
    if (selTok.includes(rec) || recTok.includes(sel)) return true;
    return recTok.length > 0 && recTok.every((t) => selTok.includes(t));
  }

  function expandCombination(name) {
    const parts = lookupComboParts(name);
    if (!parts || !parts.length) {
      const base = stripSaltName(name);
      return [{
        apcdDrugId: slug(base),
        formattedGenericName: displayName(base),
        isCombination: false,
        ingredientApcdDrugIds: [slug(base)],
        expandedAutomatically: false,
        vocabularyVersion: state.versions.apcdVocabularyVersion
      }];
    }
    return parts.map((ing) => ({
      apcdDrugId: slug(ing),
      formattedGenericName: displayName(ing),
      isCombination: true,
      sourceCombinationApcdDrugId: slug(name),
      ingredientApcdDrugIds: parts.map(slug),
      expandedAutomatically: true,
      vocabularyVersion: state.versions.apcdVocabularyVersion
    }));
  }

  function addSelection(rawName) {
    const expanded = expandCombination(rawName);
    expanded.forEach((item) => {
      if (state.selections.some((s) => s.apcdDrugId === item.apcdDrugId)) return;
      state.selections.push(item);
    });
    renderChips();
    rerenderResults();
  }

  function removeSelection(apcdDrugId) {
    const gone = state.selections.find((s) => s.apcdDrugId === apcdDrugId);
    state.selections = state.selections.filter((s) => s.apcdDrugId !== apcdDrugId);
    if (gone && gone.expandedAutomatically && gone.sourceCombinationApcdDrugId) {
      const warn = document.getElementById("pgx-combo-warning");
      if (warn) {
        warn.style.display = "";
        warn.textContent = "An automatically expanded combination ingredient was removed. The original combination is now only partially represented.";
      }
    }
    renderChips();
    rerenderResults();
  }

  function currentIngredientNames() {
    const fromChips = state.selections.map((s) => s.formattedGenericName);
    if (fromChips.length) return fromChips;
    const riskDrugs = (typeof getMultiSelectValues === "function" && typeof drugsEl !== "undefined")
      ? getMultiSelectValues(drugsEl).map(displayName)
      : [];
    return riskDrugs;
  }

  function isActionable(rec) {
    return !!(rec && ACTIONABLE_CATEGORIES[rec.actionCategory]);
  }

  function visibleRecs(recs) {
    const scope = state.drugScope;
    const names = currentIngredientNames();
    let out = recs || [];
    if (scope !== "ALL_MATCHED") {
      out = names.length
        ? out.filter((r) => names.some((n) => nameMatches(r.formattedGenericName, n)))
        : out.filter((r) => r.inRegimen);
    }
    if (state.actionableOnly) out = out.filter(isActionable);
    return out;
  }

  function classHit(name, klass) {
    const n = norm(displayName(name));
    return klass.some((k) => n === k || n.indexOf(k) !== -1);
  }

  function combinations3(items) {
    const out = [];
    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) {
        for (let k = j + 1; k < items.length; k++) {
          out.push([items[i], items[j], items[k]]);
        }
      }
    }
    return out;
  }

  function uniqueDisplayNames(names) {
    const seen = new Set();
    const out = [];
    (names || []).forEach((raw) => {
      const d = displayName(raw);
      const key = norm(d);
      if (!d || seen.has(key)) return;
      seen.add(key);
      out.push(d);
    });
    return out;
  }

  const MAX_TRIPLET_ALERTS = 250;

  function scopedTripletNames() {
    // Triplets enumerate the patient's regimen, never every CPIC-matched drug.
    // ALL_MATCHED can return 90+ generics (C(90,3) > 100k rows) and freeze the tab.
    return uniqueDisplayNames(currentIngredientNames());
  }

  function buildTripletAlerts(ingredientNames) {
    const names = uniqueDisplayNames(ingredientNames);
    if (names.length < 3) return [];
    const alerts = combinations3(names).map((picked) => {
      const cns = picked.some((p) => classHit(p, CNS_OPIOIDS))
        && picked.some((p) => classHit(p, CNS_GABAS))
        && picked.some((p) => classHit(p, CNS_BENZOS));
      const pattern = cns ? "opioid-gabapentinoid-benzodiazepine" : "three-way-match";
      return {
        ingredientApcdDrugIds: picked.map(slug),
        ingredientGenericNames: picked,
        modelVersion: state.versions.tripletModelVersion,
        thresholdVersion: state.versions.thresholdVersion,
        signalLevel: cns ? "HIGH" : "MODERATE",
        modelScore: cns ? 0.82 : 0.5,
        attribution: { [pattern]: cns ? 0.82 : 0.5 },
        mappingStatus: cns ? "VERIFIED" : "DETECTED",
        disclaimer: "Not a CPIC pharmacogenomic recommendation"
      };
    }).sort((a, b) => {
      if (a.signalLevel !== b.signalLevel) return a.signalLevel === "HIGH" ? -1 : 1;
      return a.ingredientGenericNames.join(" ").localeCompare(b.ingredientGenericNames.join(" "));
    });
    if (alerts.length > MAX_TRIPLET_ALERTS) {
      const truncated = alerts.slice(0, MAX_TRIPLET_ALERTS);
      truncated._truncatedFrom = alerts.length;
      return truncated;
    }
    return alerts;
  }

  function renderHeader(payload, calls) {
    const el = document.getElementById("pgx-session-header");
    if (!el) return;
    const assessed = calls.filter((c) => c.phenotypeConfidence !== "INDETERMINATE");
    const unresolved = calls.filter((c) => c.phenotypeConfidence === "INDETERMINATE");
    const v = state.versions;
    el.innerHTML =
      '<div class="pgx-session-grid">' +
      "<div><strong>Session</strong><br>" + (payload.patient_id ? "Pseudonym on file" : "Anonymous") + "</div>" +
      "<div><strong>Generated</strong><br>" + (payload.timestamp || "—") + "</div>" +
      "<div><strong>Input</strong><br>Consumer DNA / gene alleles</div>" +
      "<div><strong>Genes assessed</strong><br>" + assessed.map((c) => c.gene).join(", ") + "</div>" +
      "<div><strong>Unresolved</strong><br>" + (unresolved.length ? unresolved.map((c) => c.gene).join(", ") : "None") + "</div>" +
      "<div><strong>APCD vocab</strong><br>" + v.apcdVocabularyVersion + "</div>" +
      "<div><strong>CPIC bundle</strong><br>" + v.cpicKnowledgeVersion + "</div>" +
      "<div><strong>Crosswalk</strong><br>" + v.crosswalkVersion + "</div>" +
      "</div>" +
      '<p class="pgx-privacy-note">Privacy notice: raw genome files stay in this browser session and are not written to the URL or analytics. Clinical review required before any prescribing change.</p>' +
      '<p class="pgx-privacy-note">Phenotype table: star alleles are resolved on the server from official CPIC allele definitions; phenotypes come from the official diplotype→phenotype table (' +
      (v.phenotypeTableVersion || "pgx-phenotype-v1") +
      "). Unphased consumer DNA cannot uniquely phase multi-variant stars — those genes stay indeterminate. Unlisted pairs are never treated as normal. This is not a full CPIC translation service.</p>";
    renderQr();
  }

  function verificationPayload() {
    const recs = visibleRecs(state.lastRecs);
    return {
      type: "pgx-card-verify",
      v: 1,
      generated: (state.lastPayload && state.lastPayload.timestamp) || new Date().toISOString(),
      scope: state.drugScope,
      actionableOnly: !!state.actionableOnly,
      genes: (state.lastCalls || []).map((c) => ({
        gene: c.gene,
        diplotype: c.diplotype || null,
        phenotype: c.phenotype || null
      })),
      recCount: recs.length,
      actionableCount: recs.filter(isActionable).length
    };
  }

  function renderQr() {
    const canvas = document.getElementById("pgx-card-qr");
    if (!canvas || !state.lastPayload) return;
    const payload = JSON.stringify(verificationPayload());
    if (window.QRCode && typeof window.QRCode.toCanvas === "function") {
      window.QRCode.toCanvas(canvas, payload, { width: 128, margin: 1, errorCorrectionLevel: "M" }, function () {});
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#0f172a";
    ctx.font = "10px system-ui";
    ctx.fillText("QR unavailable", 18, 68);
  }

  function renderSummary(recs, alerts) {
    const el = document.getElementById("pgx-summary-cards");
    if (!el) return;
    const n = (cat) => recs.filter((r) => r.actionCategory === cat).length;
    const dose = n("DOSE_REDUCTION_OR_TITRATION") + n("DOSE_INCREASE_OR_ALTERNATIVE") + n("ENHANCED_MONITORING");
    el.innerHTML =
      cardCount("Avoid / alternative needed", n("AVOID_OR_USE_ALTERNATIVE"), "pgx-act-avoid") +
      cardCount("Dose or monitoring action", dose, "pgx-act-dose") +
      cardCount("No PGx action identified", n("STANDARD_PRESCRIBING"), "pgx-act-std") +
      cardCount("Indeterminate / unsupported", n("INSUFFICIENT_GENOTYPE_RESOLUTION") + n("NO_CPIC_RECOMMENDATION"), "pgx-act-indet") +
      cardCount("High-priority triplet alerts", alerts.filter((a) => a.signalLevel === "HIGH").length, "pgx-act-triplet") +
      cardCount("Moderate triplet alerts", alerts.filter((a) => a.signalLevel === "MODERATE").length, "pgx-act-triplet");
  }

  function cardCount(label, n, cls) {
    return '<div class="pgx-count-card ' + cls + '"><div class="pgx-count-n">' + n + "</div><div>" + label + "</div></div>";
  }

  function recText(rec, clinician) {
    if (!clinician) {
      if (rec.actionCategory === "AVOID_OR_USE_ALTERNATIVE") return "This medicine may not be a good fit based on a reviewed PGx guideline. Ask a clinician about alternatives.";
      if (rec.actionCategory.indexOf("DOSE") === 0 || rec.actionCategory === "ENHANCED_MONITORING") return "Your gene result may change how this medicine is used. A clinician should review the dose.";
      if (rec.actionCategory === "INSUFFICIENT_GENOTYPE_RESOLUTION") return "The uploaded file does not support a reliable result for this gene–drug pair.";
      return "No PGx change is identified from the current result.";
    }
    const meta = ACTION_META[rec.actionCategory] || {};
    const canned = String(rec.recommendationText || "").replace(/_/g, " ").toLowerCase();
    const richer = rec.recommendationText && canned !== String(rec.actionCategory || "").replace(/_/g, " ").toLowerCase()
      ? rec.recommendationText
      : (meta.label || rec.recommendationText || "");
    return richer + (rec.phenotype ? " (" + rec.gene + " " + rec.phenotype + ")" : "");
  }

  function renderQueue(recs) {
    const el = document.getElementById("pgx-action-queue");
    if (!el) return;
    const clinician = state.viewMode === "clinician";
    if (!recs.length) {
      el.innerHTML = '<p class="pgx-empty">No medication actions in the current drug scope. Select APCD generics or switch to All matched drugs.</p>';
      return;
    }
    el.innerHTML = recs.map((rec) => {
      const meta = ACTION_META[rec.actionCategory];
      return (
        '<article class="pgx-action-card ' + meta.cls + '" data-drug="' + rec.apcdDrugId + '">' +
        '<div class="pgx-action-kicker"><span class="pgx-evidence-badge">CPIC-guideline action</span>' +
        (rec.inRegimen ? '<span class="pgx-evidence-badge pgx-badge-active">Active / selected</span>' : "") +
        "</div>" +
        "<h3>" + rec.formattedGenericName + "</h3>" +
        '<p class="pgx-action-label">' + meta.label + "</p>" +
        (clinician
          ? "<p>Gene: " + rec.gene + " · Diplotype: " + (rec.diplotype || "—") + " · Phenotype: " + (rec.phenotype || "Indeterminate") + "</p>" +
            "<p>CPIC action: " + meta.label + (rec.cpicLevel ? " · Level " + rec.cpicLevel : "") + "</p>" +
            (rec.sourceUrl ? '<p>Evidence: <a href="' + rec.sourceUrl + '" target="_blank" rel="noopener">CPIC guideline</a> · ' + rec.cpicGuidelineVersion + "</p>" : "") +
            (rec.fdaLabel ? "<p>Supporting context: FDA label " + rec.fdaLabel + "</p>" : "") +
            (clinician ? "<p>APCD ID: " + rec.apcdDrugId + "</p>" : "")
          : "<p>" + recText(rec, false) + "</p><p>Evidence: PGx guideline reviewed</p>") +
        "</article>"
      );
    }).join("");
  }

  function renderMatrix(recs, calls) {
    const el = document.getElementById("pgx-action-matrix");
    if (!el) return;
    const drugs = [];
    const seen = new Set();
    recs.forEach((r) => {
      if (!seen.has(r.formattedGenericName)) {
        seen.add(r.formattedGenericName);
        drugs.push(r.formattedGenericName);
      }
    });
    const genes = calls.map((c) => c.gene);
    if (!drugs.length || !genes.length) {
      el.innerHTML = '<p class="pgx-empty">Matrix appears after gene calls and matched APCD drugs are available.</p>';
      return;
    }
    let html = '<table class="pgx-matrix" role="grid"><caption>Gene–drug actionability (scoped medications)</caption><thead><tr><th>Drug</th>';
    genes.forEach((g) => { html += "<th>" + g + "</th>"; });
    html += "</tr></thead><tbody>";
    drugs.forEach((drug) => {
      html += "<tr><th scope=\"row\">" + drug + "</th>";
      genes.forEach((g) => {
        const rec = recs.find((r) => r.formattedGenericName === drug && r.gene === g);
        if (!rec) {
          html += '<td class="pgx-cell-none">No guideline</td>';
        } else if (rec.actionCategory === "INSUFFICIENT_GENOTYPE_RESOLUTION") {
          html += '<td class="pgx-cell-indet">Insufficient resolution</td>';
        } else {
          html += '<td class="' + ACTION_META[rec.actionCategory].cls + '">' + ACTION_META[rec.actionCategory].label + "</td>";
        }
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    el.innerHTML = html;
  }

  function renderTriplets(alerts) {
    const el = document.getElementById("pgx-triplet-panel");
    if (!el) return;
    if (!alerts.length) {
      el.innerHTML = '<p class="pgx-empty">No three-way medication matches in the current drug scope. Select at least three APCD generics to enumerate triplets.</p>';
      return;
    }
    const truncatedFrom = alerts._truncatedFrom;
    const rows = alerts.map((a) => (
      '<tr class="' + (a.signalLevel === "HIGH" ? "pgx-triplet-high" : "pgx-triplet-mod") + '">' +
      "<td>" + a.ingredientGenericNames.join(" + ") + "</td>" +
      "<td>" + a.signalLevel + "</td>" +
      "<td>" + Object.keys(a.attribution).join(", ") + "</td>" +
      "</tr>"
    )).join("");
    el.innerHTML =
      '<p class="subtitle"><small>' +
      (truncatedFrom
        ? "Showing " + alerts.length.toLocaleString() + " of " + truncatedFrom.toLocaleString() + " three-way matches"
        : alerts.length.toLocaleString() + " three-way match" + (alerts.length === 1 ? "" : "es")) +
      " in the current scope. Separate from CPIC actions.</small></p>" +
      '<div class="pgx-triplet-scroll"><table class="pgx-matrix pgx-triplet-table">' +
      "<thead><tr><th>Medications</th><th>Signal</th><th>Pattern</th></tr></thead><tbody>" +
      rows + "</tbody></table></div>" +
      "<p><em>Not a CPIC pharmacogenomic recommendation. Clinical review required.</em></p>";
  }

  function renderChips() {
    const el = document.getElementById("pgx-selected-chips");
    if (!el) return;
    if (!state.selections.length) {
      el.innerHTML = '<span class="pgx-chip-empty">No APCD medications selected</span>';
      return;
    }
    el.innerHTML = state.selections.map((s) => (
      '<button type="button" class="pgx-chip" data-id="' + s.apcdDrugId + '" aria-label="Remove ' + s.formattedGenericName + '">' +
      s.formattedGenericName + " ×</button>"
    )).join("");
    el.querySelectorAll(".pgx-chip").forEach((btn) => {
      btn.addEventListener("click", () => removeSelection(btn.getAttribute("data-id")));
    });
  }

  function filterVocab(q) {
    const query = norm(q);
    if (query.length < 2) return [];
    return state.vocab.filter((name) => norm(name).indexOf(query) !== -1).slice(0, 20);
  }

  function bindAutocomplete() {
    const input = document.getElementById("pgx-drug-search");
    const list = document.getElementById("pgx-drug-suggest");
    if (!input || !list || input.dataset.bound) return;
    input.dataset.bound = "1";
    input.setAttribute("autocomplete", "off");
    input.addEventListener("input", () => {
      const hits = filterVocab(input.value);
      if (!hits.length) {
        list.hidden = input.value.trim().length < 2;
        list.innerHTML = input.value.trim().length < 2 ? "" : '<div class="pgx-suggest-empty">No matching medication exists in the approved Virginia APCD vocabulary.</div>';
        return;
      }
      list.hidden = false;
      list.innerHTML = hits.map((h) => '<button type="button" class="pgx-suggest-item" role="option">' + h + "</button>").join("");
      list.querySelectorAll(".pgx-suggest-item").forEach((btn) => {
        btn.addEventListener("click", () => {
          addSelection(btn.textContent);
          input.value = "";
          list.hidden = true;
        });
      });
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Escape") list.hidden = true;
      if (e.key === "Enter") {
        e.preventDefault();
        const first = list.querySelector(".pgx-suggest-item");
        if (first) first.click();
      }
    });
  }

  function rerenderResults() {
    if (!state.lastPayload) return;
    const recs = visibleRecs(state.lastRecs);
    const names = scopedTripletNames();
    const alerts = buildTripletAlerts(names);
    state.lastAlerts = alerts;
    renderHeader(state.lastPayload, state.lastCalls);
    renderSummary(recs, alerts);
    renderQueue(recs);
    renderMatrix(recs, state.lastCalls);
    renderTriplets(alerts);
  }

  function isExploratoryCall(call) {
    return !!(call && (call.analysisMode === "exploratory" || call.phenotypeConfidence === "EXPLORATORY"));
  }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderPipeline(pipeline) {
    const el = document.getElementById("pgx-pipeline-live");
    if (!el) return;
    const stages = (pipeline && pipeline.stages) || {};
    const order = [
      ["upload", "Upload and ingestion"],
      ["coordinates", "Coordinate annotation"],
      ["coverage", "Gene coverage"],
      ["pharmgkb", "PharmGKB / ClinPGx"],
      ["cpicPharmvar", "CPIC and PharmVar"],
      ["report", "Exploratory report"]
    ];
    if (!pipeline) {
      el.innerHTML = "";
      return;
    }
    el.innerHTML = '<div class="pgx-pipeline">' + order.map((pair) => {
      const stage = stages[pair[0]] || {};
      const detail = stage.coverageText || stage.note || stage.boundary || stage.engine || "";
      return '<div class="pgx-pipeline-step"><strong>' + esc(pair[1]) + '</strong> · ' +
        esc(stage.status || "—") + (detail ? " · " + esc(detail) : "") + "</div>";
    }).join("") + "</div>";
  }

  function testingNudges(payload) {
    if (!payload) return [];
    return payload.clinicalNudges || (payload.pipeline && payload.pipeline.clinicalNudges) || [];
  }

  function renderRecommendationCard(payload) {
    const card = document.getElementById("pgx-testing-card");
    const title = document.getElementById("pgx-testing-title");
    const status = document.getElementById("pgx-testing-status");
    const detail = document.getElementById("pgx-testing-detail");
    const actions = document.getElementById("pgx-testing-actions");
    const why = document.getElementById("pgx-testing-why");
    if (!card || !title || !status || !detail || !actions) return;
    card.classList.remove("pgx-test-needed", "pgx-test-clear", "pgx-test-lab");
    actions.innerHTML = "";
    const genealogyWhy = "A genealogy file lists variants that an ancestry array typed. It does not say which chromosome copy each variant is on, and it does not count extra or missing gene copies. A prescribing claim needs both, so this card cannot assign a diplotype, a metabolizer status, or a dose from that file. A site missing from the file is Data Not Present in File, not a normal allele.";
    if (!payload) {
      card.hidden = true;
      title.textContent = "Further testing recommendation";
      status.textContent = "";
      detail.textContent = "";
      if (why) why.textContent = "";
      return;
    }
    card.hidden = false;
    const nudges = testingNudges(payload);
    const exploratory = (state.lastCalls || []).some(isExploratoryCall);
    if (why) why.textContent = exploratory ? genealogyWhy : "";
    if (nudges.length) {
      const genes = nudges.map((nudge) => nudge.gene).filter(Boolean);
      card.classList.add("pgx-test-needed");
      title.textContent = "Further clinical test recommended";
      status.textContent = genes.length
        ? "Detected variant alleles in " + genes.join(", ") + "."
        : "A follow-on clinical test is required.";
      detail.textContent = (nudges[0] && nudges[0].nextStep) || "Order a CLIA/CAP-certified pharmacogenomic panel.";
      const links = ((nudges[0] && nudges[0].links) || []).map((link) => (
        '<a href="' + esc(link.url) + '" target="_blank" rel="noopener">' + esc(link.label) + "</a>"
      )).join("");
      actions.innerHTML = links +
        ' <button type="button" class="button-secondary" id="pgx-testing-pdf">Download summary for your doctor</button>';
      const pdf = document.getElementById("pgx-testing-pdf");
      if (pdf) pdf.addEventListener("click", () => exportPhysicianSummary());
      return;
    }
    if (exploratory) {
      card.classList.add("pgx-test-clear");
      title.textContent = "No further clinical test recommended";
      status.textContent = "No variant allele was detected in CYP2C19, CYP2D6, VKORC1, SLCO1B1, or HLA-B.";
      detail.textContent = "Missing sites stay Data Not Present in File. This file does not assign a diplotype or a dose.";
      return;
    }
    card.classList.add("pgx-test-lab");
    title.textContent = "Lab allele result";
    status.textContent = "This result uses the official diplotype-to-phenotype table.";
    detail.textContent = "The further-testing recommendation applies to consumer array files.";
  }

  function renderNudges(nudges) {
    const el = document.getElementById("pgx-clinical-nudges");
    if (!el) return;
    if (!nudges || !nudges.length) {
      el.innerHTML = "";
      return;
    }
    el.innerHTML = nudges.map((nudge) => {
      return '<article class="pgx-nudge">' +
        "<h3>Exploratory finding: " + esc(nudge.gene) + " gene region</h3>" +
        "<p><strong>Status:</strong> " + esc(nudge.status) + "</p>" +
        (nudge.summary ? "<p>" + esc(nudge.summary) + "</p>" : "") +
        (nudge.coverageText ? "<p>" + esc(nudge.coverageText) + "</p>" : "") +
        "</article>";
    }).join("");
  }

  function exportPhysicianSummary() {
    const payload = state.lastPayload || {};
    const findings = (payload.pipeline && payload.pipeline.findings) || payload.gene_calls || [];
    const jsPdfNs = window.jspdf || window.jsPDF;
    const Ctor = jsPdfNs && (jsPdfNs.jsPDF || jsPdfNs);
    if (!Ctor) throw new Error("PDF renderer is not loaded");
    const doc = new Ctor({ unit: "pt", format: "letter" });
    const lines = [
      "Exploratory raw-DNA summary for your clinician",
      "Genealogy files (AncestryDNA, 23andMe, MyHeritage) cannot support a prescribing claim.",
      "They list typed variants. They do not show which chromosome copy each variant is on,",
      "and they do not measure copy number. A CPIC dose needs a phased diplotype and copy-number validation.",
      "A site missing from the file is Data Not Present in File, not a normal allele.",
      "Do not start, stop, or change a medication from this summary.",
      ""
    ];
    findings.forEach((row) => {
      lines.push(row.gene + ": " + (row.summary || row.exploratorySummary || "Exploratory observation"));
      if (row.coverageText) lines.push(row.coverageText);
      (row.observedSites || []).forEach((site) => {
        if (site.variantAlleleObserved) {
          lines.push("Variant " + site.rsid + " (" + (site.genotype || "") + ") detected in raw upload file.");
        } else if (site.status === "Data Not Present in File") {
          lines.push(site.rsid + ": Data Not Present in File");
        }
      });
      (row.guidelineReferences || []).forEach((line) => lines.push(line));
      lines.push("CPIC defines clinical guidelines for this gene when phased.");
      lines.push("");
    });
    lines.push("Next step: order a CLIA/CAP-certified pharmacogenomic panel through a licensed clinician.");
    lines.push("Clinical PGx guidelines: https://cpicpgx.org/guidelines/");
    lines.push("Find a genetic counselor: https://findageneticcounselor.nsgc.org/");
    let y = 48;
    lines.forEach((line) => {
      const wrapped = doc.splitTextToSize(line, 520);
      wrapped.forEach((part) => {
        if (y > 740) { doc.addPage(); y = 48; }
        doc.text(part, 48, y);
        y += 14;
      });
    });
    downloadBlob(doc.output("blob"), "pgx-physician-summary.pdf");
  }

  function render(payload, options) {
    options = options || {};
    state.lastPayload = payload;
    const variants = options.variants || (payload.genes || []).map((g) => ({ gene: g.gene, variants: g.variants || [] }));
    state.lastCalls = payload.gene_calls && payload.gene_calls.length ? payload.gene_calls : buildGeneCalls(variants);
    const exploratoryCalls = state.lastCalls.filter(isExploratoryCall);
    const clinicalCalls = state.lastCalls.filter((call) => !isExploratoryCall(call));
    if (exploratoryCalls.length && !clinicalCalls.length) {
      state.lastRecs = [];
    } else if (payload.recommendations && payload.recommendations.length) {
      state.lastRecs = payload.recommendations.filter((rec) => clinicalCalls.some((call) => call.gene === rec.gene));
    } else {
      state.lastRecs = buildRecommendations(clinicalCalls, payload.drugs || [], currentIngredientNames());
    }
    if (payload.versions) {
      const tripletVersion = state.versions.tripletModelVersion;
      Object.assign(state.versions, payload.versions);
      if (tripletVersion) state.versions.tripletModelVersion = tripletVersion;
    }
    const box = document.getElementById("pgx-card-display");
    if (box) box.style.display = "block";
    bindAutocomplete();
    renderChips();
    renderPipeline(payload.pipeline);
    renderNudges(testingNudges(payload));
    renderRecommendationCard(payload);
    rerenderResults();
  }

  function setVocabFromMetadata(drugs) {
    const names = (drugs || []).map((d) => displayName(typeof d === "string" ? d : (d.display || d.code || d.value || "")))
      .filter(Boolean);
    Object.keys(state.combinations).forEach((k) => names.push(k));
    state.vocab = Array.from(new Set(names)).sort();
    seedVocabExtras();
  }

  function currentFilters(kind) {
    return {
      drugScope: state.drugScope,
      selectedApcdDrugIds: state.selections.map((s) => s.apcdDrugId),
      actionableOnly: !!state.actionableOnly,
      includePolypharmacy: true,
      includeTechnicalAppendix: kind === "technical"
    };
  }

  function exportBundle(kind) {
    const filters = currentFilters(kind);
    return {
      filters,
      versions: state.versions,
      generated: new Date().toISOString(),
      geneCalls: state.lastCalls,
      recommendations: visibleRecs(state.lastRecs),
      polypharmacyAlerts: filters.includePolypharmacy ? state.lastAlerts : [],
      verification: verificationPayload(),
      disclaimer: "Clinical review required. Not a substitute for CPIC guideline text or pharmacist judgment."
    };
  }

  function downloadBlob(blob, filename) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1500);
  }

  function csvEscape(value) {
    const s = value == null ? "" : String(value);
    if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  }

  function recommendationsCsv(recs) {
    const header = [
      "formattedGenericName", "apcdDrugId", "gene", "diplotype", "phenotype",
      "actionCategory", "recommendationText", "cpicLevel", "sourceUrl", "inRegimen"
    ];
    const rows = [header.join(",")];
    (recs || []).forEach((r) => {
      rows.push(header.map((k) => csvEscape(r[k])).join(","));
    });
    return rows.join("\n");
  }

  function appendixHtml(bundle) {
    const recRows = (bundle.recommendations || []).map((r) => (
      "<tr><td>" + (r.formattedGenericName || "") + "</td><td>" + (r.gene || "") +
      "</td><td>" + (r.diplotype || "") + "</td><td>" + (r.phenotype || "Indeterminate") +
      "</td><td>" + (r.actionCategory || "") + "</td><td>" + (r.cpicLevel || "") +
      "</td><td>" + (r.sourceUrl ? '<a href="' + r.sourceUrl + '">' + r.sourceUrl + "</a>" : "") + "</td></tr>"
    )).join("");
    const geneRows = (bundle.geneCalls || []).map((c) => (
      "<tr><td>" + (c.gene || "") + "</td><td>" + (c.diplotype || "") +
      "</td><td>" + (c.phenotype || "Indeterminate") + "</td><td>" +
      (c.phenotypeConfidence || "") + "</td><td>" + ((c.limitations || []).join("; ")) + "</td></tr>"
    )).join("");
    const tripRows = (bundle.polypharmacyAlerts || []).map((a) => (
      "<tr><td>" + ((a.ingredientGenericNames || []).join(" + ")) + "</td><td>" +
      (a.signalLevel || "") + "</td><td>" + (a.disclaimer || "") + "</td></tr>"
    )).join("");
    return "<!DOCTYPE html><html><head><meta charset='utf-8'><title>PGx Technical Appendix</title>" +
      "<style>body{font-family:system-ui,sans-serif;margin:1.5rem;color:#0f172a}table{border-collapse:collapse;width:100%;font-size:0.85rem}th,td{border:1px solid #cbd5e1;padding:0.4rem;text-align:left}h1{margin-top:0}</style></head><body>" +
      "<h1>PGx technical appendix</h1>" +
      "<p>This document is distinct from the current-view JSON export. It records versions, scoped recommendations, gene-call limitations, and polypharmacy alerts. Clinical review required.</p>" +
      "<h2>Filters</h2><pre>" + JSON.stringify(bundle.filters, null, 2) + "</pre>" +
      "<h2>Versions</h2><pre>" + JSON.stringify(bundle.versions, null, 2) + "</pre>" +
      "<h2>Gene calls</h2><table><thead><tr><th>Gene</th><th>Diplotype</th><th>Phenotype</th><th>Confidence</th><th>Limitations</th></tr></thead><tbody>" +
      geneRows + "</tbody></table>" +
      "<h2>Recommendations (" + (bundle.recommendations || []).length + ")</h2><table><thead><tr><th>Drug</th><th>Gene</th><th>Diplotype</th><th>Phenotype</th><th>Action</th><th>CPIC level</th><th>Guideline</th></tr></thead><tbody>" +
      recRows + "</tbody></table>" +
      "<h2>Polypharmacy alerts</h2><p>Not CPIC recommendations.</p><table><thead><tr><th>Medications</th><th>Signal</th><th>Disclaimer</th></tr></thead><tbody>" +
      tripRows + "</tbody></table>" +
      "<p><em>" + bundle.disclaimer + "</em></p></body></html>";
  }

  function clipboardText(bundle) {
    const recs = bundle.recommendations || [];
    const lines = [
      "PGx card export",
      "Scope: " + bundle.filters.drugScope + " | actionableOnly=" + bundle.filters.actionableOnly,
      "Generated: " + bundle.generated,
      "Genes: " + (bundle.geneCalls || []).map((c) => c.gene + " " + (c.diplotype || "") + " " + (c.phenotype || "indeterminate")).join("; "),
      "Recommendations:"
    ];
    recs.forEach((r) => {
      lines.push("- " + r.formattedGenericName + " | " + r.gene + " | " + (r.actionCategory || "") + " | " + (r.phenotype || "indeterminate"));
    });
    lines.push(bundle.disclaimer);
    return lines.join("\n");
  }

  function pharmacyBundle(bundle) {
    return {
      exportType: "pharmacy-handoff",
      generated: bundle.generated,
      filters: bundle.filters,
      versions: bundle.versions,
      medications: (bundle.recommendations || []).map((r) => ({
        formattedGenericName: r.formattedGenericName,
        apcdDrugId: r.apcdDrugId,
        gene: r.gene,
        diplotype: r.diplotype,
        phenotype: r.phenotype,
        actionCategory: r.actionCategory,
        actionLabel: (ACTION_META[r.actionCategory] || {}).label || r.recommendationText,
        guidelineUrl: r.sourceUrl || r.cpicGuidelineId || "",
        cpicLevel: r.cpicLevel || ""
      })),
      geneCalls: bundle.geneCalls,
      note: "No e-prescribe API. Import this file into a pharmacy workflow. Clinical review required.",
      disclaimer: bundle.disclaimer
    };
  }

  function exportPngFallback() {
    const recs = visibleRecs(state.lastRecs);
    const genes = (state.lastCalls || []).map((c) => c.gene + " " + (c.diplotype || "") + " " + (c.phenotype || "indeterminate"));
    const queueLines = recs.map((r) => {
      const meta = ACTION_META[r.actionCategory] || {};
      return (r.formattedGenericName || r.apcdDrugId || "drug") + " — " + (meta.label || r.actionCategory);
    });
    const lines = [
      "PGx card — action queue / summary",
      "Scope: " + state.drugScope + "   actionableOnly=" + !!state.actionableOnly,
      "Genes: " + (genes.join("  | ") || "none"),
      "Recommendations: " + recs.length + "   Actionable: " + recs.filter(isActionable).length,
      ""
    ].concat(queueLines.length ? queueLines : ["No medication actions in current scope."]);
    lines.push("");
    lines.push("Clinical review required. Unlisted allele pairs stay indeterminate.");
    const lineH = 22;
    const canvas = document.createElement("canvas");
    canvas.width = 1100;
    canvas.height = Math.max(640, 80 + lines.length * lineH);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 22px system-ui,sans-serif";
    ctx.fillText(lines[0], 24, 40);
    ctx.font = "14px system-ui,sans-serif";
    ctx.fillStyle = "#334155";
    for (let i = 1; i < lines.length; i++) {
      const y = 70 + (i - 1) * lineH;
      ctx.fillText(String(lines[i]).slice(0, 120), 24, y);
    }
    const qr = document.getElementById("pgx-card-qr");
    if (qr) {
      try { ctx.drawImage(qr, canvas.width - 152, 16, 128, 128); } catch (_) { /* ignore */ }
    }
    return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  }

  async function exportPngFromRenderedCard() {
    const root = document.querySelector("#pgx-card-display .pgx-card") || document.getElementById("pgx-card-display");
    if (!root || !window.html2canvas) return null;
    const display = document.getElementById("pgx-card-display");
    if (display && window.getComputedStyle(display).display === "none") return null;
    const canvas = await window.html2canvas(root, {
      backgroundColor: "#ffffff",
      scale: 2,
      useCORS: true,
      logging: false,
      windowWidth: Math.max(root.scrollWidth, 900),
      onclone: (doc) => {
        const clonedDisplay = doc.getElementById("pgx-card-display");
        if (clonedDisplay) clonedDisplay.style.display = "block";
        const trip = doc.querySelector(".pgx-triplet-scroll");
        if (trip) trip.style.maxHeight = "none";
      }
    });
    return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  }

  async function exportPng() {
    let blob = null;
    try {
      blob = await exportPngFromRenderedCard();
    } catch (err) {
      console.warn("html2canvas card capture failed; using stacked fallback", err);
    }
    if (!blob) blob = await exportPngFallback();
    if (!blob) throw new Error("PNG export failed");
    downloadBlob(blob, "pgx-report-current.png");
  }

  function exportPdf(bundle) {
    const jsPdfNs = window.jspdf || window.jsPDF;
    const Ctor = jsPdfNs && (jsPdfNs.jsPDF || jsPdfNs);
    if (!Ctor) throw new Error("PDF renderer is not loaded");
    const doc = new Ctor({ unit: "pt", format: "letter" });
    const lines = [
      "PGx card report",
      "Generated: " + bundle.generated,
      "Scope: " + bundle.filters.drugScope + " | actionableOnly=" + bundle.filters.actionableOnly,
      ""
    ];
    (bundle.geneCalls || []).forEach((c) => {
      lines.push("Gene " + c.gene + " " + (c.diplotype || "") + " " + (c.phenotype || "Indeterminate"));
    });
    lines.push("");
    (bundle.recommendations || []).forEach((r) => {
      lines.push((r.formattedGenericName || "") + " — " + ((ACTION_META[r.actionCategory] || {}).label || r.actionCategory));
    });
    lines.push("");
    lines.push(bundle.disclaimer);
    let y = 48;
    lines.forEach((line) => {
      const wrapped = doc.splitTextToSize(line, 520);
      wrapped.forEach((w) => {
        if (y > 740) { doc.addPage(); y = 48; }
        doc.text(w, 48, y);
        y += 14;
      });
    });
    downloadBlob(doc.output("blob"), "pgx-report-current.pdf");
  }

  async function exportCurrent(kind) {
    if (!state.lastPayload && kind !== "print") return;
    const bundle = exportBundle(kind);
    if (kind === "json") {
      downloadBlob(new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" }), "pgx-report-current.json");
      return;
    }
    if (kind === "technical") {
      downloadBlob(new Blob([appendixHtml(bundle)], { type: "text/html;charset=utf-8" }), "pgx-report-appendix.html");
      return;
    }
    if (kind === "csv") {
      downloadBlob(new Blob([recommendationsCsv(bundle.recommendations)], { type: "text/csv;charset=utf-8" }), "pgx-report-current.csv");
      return;
    }
    if (kind === "clipboard") {
      const text = clipboardText(bundle);
      if (navigator.clipboard && navigator.clipboard.writeText) await navigator.clipboard.writeText(text);
      else {
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      return;
    }
    if (kind === "pharmacy") {
      const handoff = pharmacyBundle(bundle);
      downloadBlob(new Blob([JSON.stringify(handoff, null, 2)], { type: "application/json" }), "pgx-report-pharmacy.json");
      return;
    }
    if (kind === "png") {
      await exportPng();
      return;
    }
    if (kind === "pdf") {
      exportPdf(bundle);
      return;
    }
    if (kind === "physician") {
      exportPhysicianSummary();
      return;
    }
    window.print();
  }

  async function init() {
    if (state._inited) return;
    if (!document.getElementById("pgx-drug-scope")) return;
    state._inited = true;
    try {
      const [ver, combo] = await Promise.all([
        fetch("data/pgx_reference_versions.json").then((r) => r.ok ? r.json() : {}),
        fetch("data/apcd_combination_map.json").then((r) => r.ok ? r.json() : {})
      ]);
      Object.assign(state.versions, ver);
      state.combinations = combo.combinations || {};
      state.salts = combo.salts || {};
      state.aliases = combo.aliases || {};
      if (combo.mappingVersion) state.versions.crosswalkVersion = combo.mappingVersion;
    } catch (_) { /* keep defaults */ }
    const scope = document.getElementById("pgx-drug-scope");
    if (scope) {
      scope.value = state.drugScope;
      scope.addEventListener("change", () => {
        state.drugScope = scope.value;
        rerenderResults();
      });
    }
    const actionable = document.getElementById("pgx-actionable-only");
    if (actionable) {
      actionable.checked = !!state.actionableOnly;
      actionable.addEventListener("change", () => {
        state.actionableOnly = !!actionable.checked;
        rerenderResults();
      });
    }
    document.querySelectorAll("[name='pgx-view-mode']").forEach((el) => {
      el.addEventListener("change", () => {
        state.viewMode = el.value;
        rerenderResults();
      });
    });
    const bindExport = (id, kind) => {
      document.getElementById(id)?.addEventListener("click", () => {
        Promise.resolve(exportCurrent(kind)).catch((err) => {
          const status = document.getElementById("pgx-status");
          if (status) {
            status.style.display = "";
            status.textContent = "Export failed: " + (err && err.message ? err.message : err);
            status.className = "status-message error";
          }
        });
      });
    };
    bindExport("pgx-export-current", "print");
    bindExport("pgx-export-json", "json");
    bindExport("pgx-export-tech", "technical");
    bindExport("pgx-export-csv", "csv");
    bindExport("pgx-export-png", "png");
    bindExport("pgx-export-pdf", "pdf");
    bindExport("pgx-export-physician", "physician");
    bindExport("pgx-export-clipboard", "clipboard");
    bindExport("pgx-export-pharmacy", "pharmacy");
    seedVocabExtras();
    bindAutocomplete();
    renderChips();
  }

  function reset() {
    state.selections = [];
    state.drugScope = "ACTIVE";
    state.viewMode = "clinician";
    state.actionableOnly = false;
    state.lastPayload = null;
    state.lastCalls = [];
    state.lastRecs = [];
    state.lastAlerts = [];
    const scope = document.getElementById("pgx-drug-scope");
    if (scope) scope.value = "ACTIVE";
    const actionable = document.getElementById("pgx-actionable-only");
    if (actionable) actionable.checked = false;
    document.querySelectorAll("[name='pgx-view-mode']").forEach((el) => {
      el.checked = el.value === "clinician";
    });
    const warn = document.getElementById("pgx-combo-warning");
    if (warn) {
      warn.style.display = "none";
      warn.textContent = "";
    }
    const box = document.getElementById("pgx-card-display");
    if (box) box.style.display = "none";
    const queue = document.getElementById("pgx-action-queue");
    if (queue) queue.innerHTML = "";
    const matrix = document.getElementById("pgx-action-matrix");
    if (matrix) matrix.innerHTML = "";
    const trips = document.getElementById("pgx-triplet-panel");
    if (trips) trips.innerHTML = "";
    const summary = document.getElementById("pgx-summary-cards");
    if (summary) summary.innerHTML = "";
    const genes = document.getElementById("pgx-genes-list");
    if (genes) genes.innerHTML = "";
    const details = document.getElementById("pgx-gene-details");
    if (details) details.innerHTML = "";
    const header = document.getElementById("pgx-session-header");
    if (header) header.innerHTML = "";
    const pid = document.getElementById("pgx-patient-id");
    if (pid) pid.textContent = "";
    const qr = document.getElementById("pgx-card-qr");
    if (qr && qr.getContext) {
      const ctx = qr.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, qr.width, qr.height);
    }
    const nudges = document.getElementById("pgx-clinical-nudges");
    if (nudges) nudges.innerHTML = "";
    const pipelineLive = document.getElementById("pgx-pipeline-live");
    if (pipelineLive) pipelineLive.innerHTML = "";
    renderRecommendationCard(null);
    renderChips();
  }

  global.PgxWorkflow = {
    init,
    render,
    reset,
    setVocabFromMetadata,
    addSelection,
    currentIngredientNames,
    buildGeneCalls,
    exportCurrent,
    isActionable,
    ACTION_META,
    state
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(window);
