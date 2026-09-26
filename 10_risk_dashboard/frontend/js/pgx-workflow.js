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
      exportRendererVersion: "pgx-card-export-v1"
    },
    combinations: {},
    vocab: [],
    selections: [],
    drugScope: "ACTIVE",
    viewMode: "clinician",
    lastPayload: null,
    lastCalls: [],
    lastRecs: [],
    lastAlerts: []
  };

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
        inRegimen: selected.size ? selected.has(norm(name)) : false
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

  function expandCombination(name) {
    const key = norm(displayName(name));
    const parts = state.combinations[key] || state.combinations[displayName(name)];
    if (!parts || !parts.length) {
      return [{
        apcdDrugId: slug(name),
        formattedGenericName: displayName(name),
        isCombination: false,
        ingredientApcdDrugIds: [slug(name)],
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

  function visibleRecs(recs) {
    const scope = state.drugScope;
    const names = new Set(currentIngredientNames().map(norm));
    if (scope === "ALL_MATCHED") return recs;
    if (!names.size) return recs.filter((r) => r.inRegimen);
    return recs.filter((r) => names.has(norm(r.formattedGenericName)));
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

  function scopedTripletNames() {
    if (state.drugScope === "ALL_MATCHED" && state.lastRecs && state.lastRecs.length) {
      return uniqueDisplayNames(state.lastRecs.map((r) => r.formattedGenericName));
    }
    return uniqueDisplayNames(currentIngredientNames());
  }

  function buildTripletAlerts(ingredientNames) {
    const names = uniqueDisplayNames(ingredientNames);
    if (names.length < 3) return [];
    return combinations3(names).map((picked) => {
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
      '<p class="pgx-privacy-note">Privacy notice: raw genome files stay in this browser session and are not written to the URL or analytics. Clinical review required before any prescribing change.</p>';
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
    return rec.recommendationText + (rec.phenotype ? " (" + rec.gene + " " + rec.phenotype + ")" : "");
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
      el.innerHTML = '<p class="pgx-empty">No three-way medication matches in the current drug scope.</p>';
      return;
    }
    const rows = alerts.map((a) => (
      '<tr class="' + (a.signalLevel === "HIGH" ? "pgx-triplet-high" : "pgx-triplet-mod") + '">' +
      "<td>" + a.ingredientGenericNames.join(" + ") + "</td>" +
      "<td>" + a.signalLevel + "</td>" +
      "<td>" + Object.keys(a.attribution).join(", ") + "</td>" +
      "</tr>"
    )).join("");
    el.innerHTML =
      '<p class="subtitle"><small>' + alerts.length.toLocaleString() +
      " three-way match" + (alerts.length === 1 ? "" : "es") +
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

  function render(payload, options) {
    options = options || {};
    state.lastPayload = payload;
    const variants = options.variants || (payload.genes || []).map((g) => ({ gene: g.gene, variants: g.variants || [] }));
    state.lastCalls = payload.gene_calls && payload.gene_calls.length ? payload.gene_calls : buildGeneCalls(variants);
    state.lastRecs = payload.recommendations && payload.recommendations.length
      ? payload.recommendations
      : buildRecommendations(state.lastCalls, payload.drugs || [], currentIngredientNames());
    if (payload.versions) {
      const tripletVersion = state.versions.tripletModelVersion;
      Object.assign(state.versions, payload.versions);
      if (tripletVersion) state.versions.tripletModelVersion = tripletVersion;
    }
    const box = document.getElementById("pgx-card-display");
    if (box) box.style.display = "block";
    bindAutocomplete();
    renderChips();
    rerenderResults();
  }

  function setVocabFromMetadata(drugs) {
    const names = (drugs || []).map((d) => displayName(typeof d === "string" ? d : (d.display || d.code || d.value || "")))
      .filter(Boolean);
    Object.keys(state.combinations).forEach((k) => names.push(k));
    state.vocab = Array.from(new Set(names)).sort();
    seedVocabExtras();
  }

  function exportCurrent(kind) {
    const filters = {
      drugScope: state.drugScope,
      selectedApcdDrugIds: state.selections.map((s) => s.apcdDrugId),
      actionableOnly: false,
      includePolypharmacy: true,
      includeTechnicalAppendix: kind === "technical"
    };
    const recs = visibleRecs(state.lastRecs);
    const blob = {
      filters,
      versions: state.versions,
      generated: new Date().toISOString(),
      geneCalls: state.lastCalls,
      recommendations: recs,
      polypharmacyAlerts: filters.includePolypharmacy ? state.lastAlerts : [],
      disclaimer: "Clinical review required. Not a substitute for CPIC guideline text or pharmacist judgment."
    };
    if (kind === "json" || kind === "technical") {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([JSON.stringify(blob, null, 2)], { type: "application/json" }));
      a.download = "pgx-report-" + (kind === "technical" ? "appendix" : "current") + ".json";
      a.click();
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
    } catch (_) { /* keep defaults */ }
    const scope = document.getElementById("pgx-drug-scope");
    if (scope) scope.addEventListener("change", () => {
      state.drugScope = scope.value;
      rerenderResults();
    });
    document.querySelectorAll("[name='pgx-view-mode']").forEach((el) => {
      el.addEventListener("change", () => {
        state.viewMode = el.value;
        rerenderResults();
      });
    });
    document.getElementById("pgx-export-current")?.addEventListener("click", () => exportCurrent("print"));
    document.getElementById("pgx-export-json")?.addEventListener("click", () => exportCurrent("json"));
    document.getElementById("pgx-export-tech")?.addEventListener("click", () => exportCurrent("technical"));
    seedVocabExtras();
    bindAutocomplete();
    renderChips();
  }

  global.PgxWorkflow = {
    init,
    render,
    setVocabFromMetadata,
    addSelection,
    currentIngredientNames,
    buildGeneCalls,
    ACTION_META,
    state
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})(window);
