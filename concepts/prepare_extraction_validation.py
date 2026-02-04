#!/usr/bin/env python3
"""
Prepare data for Reader Study 2: LLM extraction fidelity validation.

Samples reports, pairs them with Ministral-8B-extracted observations, and
outputs a CSV for radiologist validation of extraction accuracy.

Two modes:
  1) --from-json : Parse existing raw extraction output (data/mimic_concepts.json)
  2) default     : Re-extract from a sample of reports using vLLM + Ministral-8B

Usage (on GPU server):
    # Mode 1: from existing JSON
    python prepare_extraction_validation.py --from-json

    # Mode 2: re-extract a fresh sample
    python prepare_extraction_validation.py

Requirements: pandas, numpy.  Mode 2 additionally needs: vllm
"""

import os
import json
import ast
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS — adjust as needed or override with env vars
# ============================================================
CXR_CONCEPT_ROOT = os.environ.get(
    "CXR_CONCEPT_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent of concepts/
)
IMPRESSIONS_CSV = os.path.join(CXR_CONCEPT_ROOT, "data", "mimic_impressions.csv")
RAW_JSON = os.path.join(CXR_CONCEPT_ROOT, "concepts", "data", "mimic_concepts.json")
REPORT_DIR = os.environ.get(
    "MIMIC_REPORT_DIR",
    "/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0/reports/files/"
)
CXR_PATHS_CSV = os.path.join(CXR_CONCEPT_ROOT, "data", "cxr_paths.csv")
OUTPUT_DIR = os.path.join(CXR_CONCEPT_ROOT, "concepts", "results", "extraction_validation")

N_REPORTS = 50          # number of reports to sample
N_PAIRS = 100           # target number of report-observation pairs
NEGATION_FRACTION = 0.3 # fraction of pairs with negation
SEED = 42

NEGATION_WORDS = [
    "no ", "no\n", "not ", "without ", "absent", "negative", "unremarkable",
    "normal", "clear", "none ", "denied", "resolved", "removed",
]

# Same prompt template used in get_concepts.py
EXTRACTION_PROMPT = """
You are a helpful assistant. Please respond in valid JSON only.
Question: What are the descriptive observations in the report? FINAL REPORT EXAMINATION: CHEST (PORTABLE AP)CHEST (PORTABLE AP)i INDICATION: ___ year old woman with SOB // eval for sign of PNA, effusion, pulm vascular congestion COMPARISON: Chest radiographs ___ IMPRESSION: Mild pulmonary edema and small to moderate bilateral pleural effusions all improved since ___ following extubation. Heart size normal. No pneumothorax. Left subclavian line ends in the SVC
Respond with a JSON object with this structure:
{{
  "observations": ["Mild pulmonary edema", "Small to moderate bilateral pleural effusions", "Heart size normal", "No pneumothorax", "Left subclavian line ends in the SVC"],
}}
Question: What are the descriptive observations in the report? FINAL REPORT EXAMINATION: CHEST (AP AND LAT) INDICATION: History: ___M with cough, fever TECHNIQUE: Upright AP and lateral views of the chest COMPARISON: Chest CT ___ FINDINGS: Cardiac silhouette size is normal. Mediastinal and hilar contours are unremarkable. Lungs are hyperinflated. No pulmonary edema is seen. Ill-defined patchy opacities are noted in the left lung base, concerning for pneumonia. Blunting of the costophrenic angles bilaterally suggests trace bilateral pleural effusions, more pronounced on the left. No pneumothorax is present. No acute osseous abnormalities detected. Multiple clips are again noted at the gastroesophageal junction and in the right upper quadrant of the abdomen. IMPRESSION: Patchy ill-defined left basilar opacity concerning for pneumonia. Small bilateral pleural effusions.
Respond with a JSON object with this structure:
{{
  "observations": ["Cardiac silhouette size is normal", "Mediastinal and hilar contours are unremarkable", "Lungs are hyperinflated", "No pulmonary edema is seen", "Ill-defined patchy opacities in the left lung base", "Blunting of the costophrenic angles bilaterally", "No pneumothorax", "No acute osseous abnormalities", "Small bilateral pleural effusions"],
}}
Question: What are the descriptive observations in the report? {report_text}
Respond with a JSON object with this structure:
"""


def has_negation(text):
    """Check whether an observation contains negation language."""
    lower = text.lower()
    return any(w in lower for w in NEGATION_WORDS)


def parse_raw_json(json_path):
    """Parse the raw extraction JSON from get_concepts.py output."""
    print(f"Loading raw extraction JSON: {json_path}")
    with open(json_path, "r") as f:
        results = json.load(f)

    records = []
    for entry in results:
        report_id = entry["id"]
        # Extract study_id from path like .../s56789012.txt
        study_id = Path(report_id).stem  # e.g. "s56789012"
        try:
            parsed = ast.literal_eval(entry["model_output"])
            observations = parsed.get("observations", [])
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(entry["model_output"])
                observations = parsed.get("observations", [])
            except json.JSONDecodeError:
                continue

        for obs in observations:
            obs_clean = obs.strip()
            if obs_clean:
                records.append({
                    "study_id": study_id,
                    "report_path": report_id,
                    "extracted_observation": obs_clean,
                    "has_negation": has_negation(obs_clean),
                })
    print(f"  Parsed {len(records)} observation-report pairs from {len(results)} reports")
    return pd.DataFrame(records)


def extract_from_sample(impressions_df, n_reports, seed):
    """Re-extract observations from a sample of reports using vLLM + Ministral-8B."""
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    rng = np.random.RandomState(seed)
    # Sample reports with non-empty impressions
    valid = impressions_df[impressions_df["impression"].str.len() > 20].copy()
    sampled = valid.sample(n=min(n_reports, len(valid)), random_state=rng)

    model_name = "mistralai/Ministral-8B-Instruct-2410"
    sampling_params = SamplingParams(max_tokens=8192, temperature=0, top_k=-1)
    llm = LLM(model=model_name, tokenizer_mode="mistral",
              config_format="mistral", load_format="mistral")

    records = []
    for _, row in sampled.iterrows():
        study_id = Path(row["filename"]).stem
        report_text = " ".join(row["impression"].split())

        prompt = EXTRACTION_PROMPT.format(report_text=report_text)
        messages = [{"role": "user", "content": prompt}]
        outputs = llm.chat(messages, sampling_params=sampling_params)
        raw_output = outputs[0].outputs[0].text

        try:
            parsed = ast.literal_eval(raw_output)
            observations = parsed.get("observations", [])
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(raw_output)
                observations = parsed.get("observations", [])
            except json.JSONDecodeError:
                observations = []

        for obs in observations:
            obs_clean = obs.strip()
            if obs_clean:
                records.append({
                    "study_id": study_id,
                    "original_text": report_text,
                    "extracted_observation": obs_clean,
                    "has_negation": has_negation(obs_clean),
                })

    print(f"  Extracted {len(records)} observations from {len(sampled)} reports")
    return pd.DataFrame(records)


def attach_impression_text(pairs_df, impressions_df):
    """Attach original impression text to pairs based on study_id."""
    imp_map = {}
    for _, row in impressions_df.iterrows():
        sid = Path(row["filename"]).stem
        imp_map[sid] = row["impression"]

    pairs_df["original_text"] = pairs_df["study_id"].map(imp_map)
    before = len(pairs_df)
    pairs_df = pairs_df.dropna(subset=["original_text"])
    if len(pairs_df) < before:
        print(f"  Dropped {before - len(pairs_df)} pairs with no matching impression")
    return pairs_df


def stratified_sample(pairs_df, n_pairs, negation_frac, seed):
    """Sample n_pairs with stratification for negation content."""
    rng = np.random.RandomState(seed)
    neg = pairs_df[pairs_df["has_negation"]]
    pos = pairs_df[~pairs_df["has_negation"]]

    n_neg = min(int(n_pairs * negation_frac), len(neg))
    n_pos = min(n_pairs - n_neg, len(pos))
    # Backfill if one stratum is too small
    if n_neg + n_pos < n_pairs:
        shortfall = n_pairs - n_neg - n_pos
        if len(neg) > n_neg:
            n_neg = min(n_neg + shortfall, len(neg))
        else:
            n_pos = min(n_pos + shortfall, len(pos))

    sampled_neg = neg.sample(n=n_neg, random_state=rng) if n_neg > 0 else neg.iloc[:0]
    sampled_pos = pos.sample(n=n_pos, random_state=rng) if n_pos > 0 else pos.iloc[:0]
    sampled = pd.concat([sampled_neg, sampled_pos]).sample(frac=1, random_state=rng)
    sampled = sampled.reset_index(drop=True)
    sampled.index.name = "pair_id"
    sampled = sampled.reset_index()
    print(f"  Sampled {len(sampled)} pairs ({n_neg} with negation, {n_pos} without)")
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Prepare extraction fidelity validation data")
    parser.add_argument("--from-json", action="store_true",
                        help="Parse existing raw extraction JSON instead of re-extracting")
    parser.add_argument("--json-path", type=str, default=RAW_JSON,
                        help="Path to raw extraction JSON")
    parser.add_argument("--n-reports", type=int, default=N_REPORTS)
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load impressions
    print(f"Loading impressions: {IMPRESSIONS_CSV}")
    impressions_df = pd.read_csv(IMPRESSIONS_CSV)
    impressions_df["impression"] = impressions_df["impression"].fillna("")
    print(f"  {len(impressions_df)} reports")

    # Get report-observation pairs
    if args.from_json:
        if not os.path.exists(args.json_path):
            raise FileNotFoundError(
                f"Raw JSON not found: {args.json_path}\n"
                "Run without --from-json to re-extract, or provide --json-path"
            )
        pairs_df = parse_raw_json(args.json_path)
        pairs_df = attach_impression_text(pairs_df, impressions_df)
    else:
        print(f"Re-extracting from {args.n_reports} sampled reports using Ministral-8B...")
        pairs_df = extract_from_sample(impressions_df, args.n_reports, args.seed)

    # Stratified sampling
    sampled = stratified_sample(pairs_df, args.n_pairs, NEGATION_FRACTION, args.seed)

    # Select and order output columns
    out_cols = ["pair_id", "study_id", "original_text", "extracted_observation", "has_negation"]
    sampled = sampled[out_cols]

    # Save
    out_path = os.path.join(args.output_dir, "extraction_validation_pairs.csv")
    sampled.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  {len(sampled)} pairs from {sampled['study_id'].nunique()} unique reports")
    print(f"  Negation pairs: {sampled['has_negation'].sum()}")
    print(f"  Non-negation pairs: {(~sampled['has_negation']).sum()}")
    print("Done.")


if __name__ == "__main__":
    main()
