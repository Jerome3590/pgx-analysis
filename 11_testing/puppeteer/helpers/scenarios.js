"use strict";

/**
 * Shared code pools and scenario definitions.
 * Mirrors test_combinatorial_risk.py — same pools, same density-tier counts.
 *
 * Default thresholds: p25=5, p50=15, p95=50
 *   baseline : 0 codes  → is_baseline=true
 *   low      : 3 codes  → n_event_bin=low    (≤ p25)
 *   medium   : 10 codes → n_event_bin=medium
 *   high     : 25 codes → n_event_bin=high
 *   extreme  : 55 codes → n_event_bin=extreme (> p95)
 */

const OPD_DRUGS = [
  "oxycodone", "hydrocodone", "tramadol", "gabapentin", "alprazolam",
  "cyclobenzaprine", "fentanyl", "codeine", "methadone", "morphine",
  "diazepam", "clonazepam", "buprenorphine", "oxymorphone", "hydromorphone",
  "carisoprodol", "zolpidem", "lorazepam", "pregabalin", "duloxetine",
];
const OPD_ICDS = [
  "M54.5", "G89.29", "F41.1", "F32.1", "F17.210", "R51", "M25.511",
  "Z87.891", "J06.9", "M54.41", "G89.4", "M79.3", "F33.1",
  "M54.16", "G89.11", "F41.0", "M47.816", "Z79.891", "G89.21", "M54.50",
];
const OPD_CPTS = [
  "99213", "80305", "99396", "99214", "99203", "80306", "97110",
  "97012", "90832", "90834", "72100", "72148", "73560", "99215", "99204",
];

const NON_DRUGS = [
  "furosemide", "hydrochlorothiazide", "lisinopril", "metformin", "simvastatin",
  "atorvastatin", "metoprolol", "amlodipine", "carvedilol", "losartan",
  "warfarin", "aspirin", "omeprazole", "levothyroxine", "albuterol",
  "prednisone", "levofloxacin", "alprazolam", "lorazepam", "acetaminophen",
];
const NON_ICDS = [
  "I10", "E11.9", "E78.5", "I50.9", "N18.3", "I25.10", "J44.1",
  "E03.9", "G47.33", "M79.3", "K21.0", "D64.9", "F03.90", "G20",
  "I48.91", "I63.9", "N39.0", "R06.09", "Z79.01", "M17.11",
];
const NON_CPTS = [
  "99213", "99214", "93000", "83036", "85025", "80053", "36415",
  "99396", "93306", "71046", "93010", "82947", "84443", "86900", "99395",
];

const POOLS = {
  opioid_ed:     { drugs: OPD_DRUGS, icds: OPD_ICDS, cpts: OPD_CPTS },
  non_opioid_ed: { drugs: NON_DRUGS, icds: NON_ICDS, cpts: NON_CPTS },
};

function makeScenarios(cohort) {
  const { drugs, icds, cpts } = POOLS[cohort];
  return {
    baseline: { drugs: [],              icds: [],              cpts: [],              expectedBin: null,      totalCodes: 0  },
    low:      { drugs: drugs.slice(0,1), icds: icds.slice(0,1), cpts: cpts.slice(0,1), expectedBin: "low",     totalCodes: 3  },
    medium:   { drugs: drugs.slice(0,4), icds: icds.slice(0,4), cpts: cpts.slice(0,2), expectedBin: "medium",  totalCodes: 10 },
    high:     { drugs: drugs.slice(0,10), icds: icds.slice(0,10), cpts: cpts.slice(0,5), expectedBin: "high",  totalCodes: 25 },
    extreme:  { drugs: drugs.slice(0,20), icds: icds.slice(0,20), cpts: cpts.slice(0,15), expectedBin: "extreme", totalCodes: 55 },
  };
}

/** Map each age band to a representative numeric age for the UI's age input. */
const AGE_BAND_MIDPOINTS = {
  "13-24":  18,
  "25-44":  35,
  "45-54":  50,
  "55-64":  60,
  "65-74":  70,
  "75-84":  80,
  "85-114": 90,
};

const COHORTS    = ["opioid_ed", "non_opioid_ed"];
const AGE_BANDS  = Object.keys(AGE_BAND_MIDPOINTS); // 0-12 excluded (UI blocks it)
const VALID_BINS = new Set(["low", "medium", "high", "extreme"]);
const VALID_BANDS = new Set(["low", "medium", "high"]);

module.exports = { makeScenarios, AGE_BAND_MIDPOINTS, COHORTS, AGE_BANDS, VALID_BINS, VALID_BANDS };
