#!/usr/bin/env python3
"""
Side-by-side comparison: On-device NLU engine vs Dialogflow CX/ES.

Usage (ES):
    python scripts/compare_dialogflow.py
        --input data/semantic_holdout_100.csv
        --project YOUR_GCP_PROJECT_ID
        --output results.csv

Usage (CX):
    python scripts/compare_dialogflow.py
        --input data/semantic_holdout_100.csv
        --project YOUR_GCP_PROJECT_ID
        --agent   YOUR_DIALOGFLOW_AGENT_ID
        --location global

The CSV must have at minimum an "utterance" column.
An "expected_intent" column is optional -- if present, both engines are
scored against it and a summary accuracy table is printed.

Dialogflow flavour is auto-detected:
  - If --agent is provided -> Dialogflow CX  (pip install google-cloud-dialogflow-cx)
  - Otherwise              -> Dialogflow ES  (pip install google-cloud-dialogflow)

Authentication:
  Option A: gcloud auth application-default login
  Option B: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
"""

import argparse
import sys
import uuid
import time
from pathlib import Path

import pandas as pd

# ── on-device NLU ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from nlu.engine import NLUEngine

# ── colours ───────────────────────────────────────────────────────────────────
GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

MATCH    = f"{GREEN}OK{RESET}"
MISMATCH = f"{RED}XX{RESET}"
SAME     = f"{CYAN}=={RESET}"


# ── Dialogflow helpers ────────────────────────────────────────────────────────

def _cx_detect(client, session_path, text, language):
    from google.cloud.dialogflowcx_v3 import TextInput, QueryInput, DetectIntentRequest
    request = DetectIntentRequest(
        session=session_path,
        query_input=QueryInput(text=TextInput(text=text), language_code=language),
    )
    response = client.detect_intent(request=request)
    qr = response.query_result
    intent_name = qr.match.intent.display_name if qr.match.intent else "Default Fallback Intent"
    return {"intent": intent_name, "confidence": round(float(qr.match.confidence), 3)}


def _es_detect(client, session_path, text, language):
    from google.cloud.dialogflow_v2 import TextInput, QueryInput
    query_input = QueryInput(text=TextInput(text=text, language_code=language))
    response = client.detect_intent(session=session_path, query_input=query_input)
    qr = response.query_result
    return {
        "intent":     qr.intent.display_name or "Default Fallback Intent",
        "confidence": round(float(qr.intent_detection_confidence), 3),
    }


def build_dialogflow_fn(project, agent, location, language):
    if agent:
        try:
            from google.cloud.dialogflowcx_v3 import SessionsClient
        except ImportError:
            sys.exit("Install CX client: pip install google-cloud-dialogflow-cx")
        endpoint = f"{location}-dialogflow.googleapis.com" if location != "global" else None
        client = SessionsClient(client_options={"api_endpoint": endpoint} if endpoint else {})

        def df_fn(utterance, session_id):
            session_path = client.session_path(project, location, agent, session_id)
            return _cx_detect(client, session_path, utterance, language)
    else:
        try:
            from google.cloud.dialogflow_v2 import SessionsClient
        except ImportError:
            sys.exit("Install ES client: pip install google-cloud-dialogflow")
        client = SessionsClient()

        def df_fn(utterance, session_id):
            session_path = client.session_path(project, session_id)
            return _es_detect(client, session_path, utterance, language)

    return df_fn


# ── on-device NLU helper ──────────────────────────────────────────────────────

def nlu_classify(engine, utterance):
    session_id = f"cmp-{uuid.uuid4().hex[:8]}"
    r = engine.handle(session_id, utterance)
    engine.reset(session_id)
    intent = r.intent if r.type != "FALLBACK" else "Default Fallback Intent"
    # semantic_rescue was added in adv-semantic branch; fall back gracefully
    semantic = getattr(r, "semantic_rescue", False)
    stage = "semantic" if semantic else ("genai" if r.type == "FALLBACK" else "tfidf")
    return {"intent": intent, "confidence": round(r.confidence, 3), "stage": stage}


# ── formatting ────────────────────────────────────────────────────────────────

def _col(text, expected, actual):
    if expected is None:
        return text
    return f"{GREEN}{text}{RESET}" if actual == expected else f"{RED}{text}{RESET}"


def print_row(i, utterance, nlu, df, expected):
    agree_sym = SAME if nlu["intent"] == df["intent"] else " "
    nlu_ok = (nlu["intent"] == expected) if expected else None
    df_ok  = (df["intent"]  == expected) if expected else None
    nlu_mark = (MATCH if nlu_ok else MISMATCH) if expected else ""
    df_mark  = (MATCH if df_ok  else MISMATCH) if expected else ""

    print(f"\n{BOLD}[{i:>3}]{RESET} {utterance}")
    if expected:
        print(f"       {DIM}expected : {expected}{RESET}")
    print(f"  On-device  {nlu_mark}  {_col(nlu['intent'], expected, nlu['intent']):<35}"
          f"  conf={nlu['confidence']:.2f}  [{nlu['stage']}]")
    print(f"  Dialogflow {df_mark}  {_col(df['intent'], expected, df['intent']):<35}"
          f"  conf={df['confidence']:.2f}  {agree_sym}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare on-device NLU vs Dialogflow")
    parser.add_argument("--input",          required=True,     help="CSV with 'utterance' column")
    parser.add_argument("--project",        required=True,     help="GCP project ID")
    parser.add_argument("--agent",          default=None,      help="Dialogflow CX agent ID (omit for ES)")
    parser.add_argument("--location",       default="global",  help="CX location (default: global)")
    parser.add_argument("--language",       default="en",      help="Language code (default: en)")
    parser.add_argument("--session-prefix", default="cmp",     help="Session ID prefix")
    parser.add_argument("--output",         default=None,      help="Save full results to CSV")
    parser.add_argument("--delay",          type=float, default=0.1,
                        help="Seconds between Dialogflow calls (default: 0.1)")
    parser.add_argument("--limit",          type=int,   default=None,
                        help="Only process first N rows")
    args = parser.parse_args()

    df_input = pd.read_csv(args.input)
    if "utterance" not in df_input.columns:
        sys.exit("CSV must have an 'utterance' column")
    has_expected = "expected_intent" in df_input.columns
    if args.limit:
        df_input = df_input.head(args.limit)

    print(f"\n{BOLD}Comparing {len(df_input)} utterances{RESET}")
    print(f"  Input      : {args.input}")
    print(f"  Dialogflow : {'CX' if args.agent else 'ES'}  project={args.project}"
          + (f"  agent={args.agent}" if args.agent else ""))
    print(f"  Has expected: {has_expected}")

    print(f"\n{DIM}Loading on-device NLU...{RESET}")
    engine = NLUEngine()
    df_fn  = build_dialogflow_fn(args.project, args.agent, args.location, args.language)

    rows = []
    for i, row in enumerate(df_input.itertuples(), 1):
        utterance = row.utterance
        expected  = getattr(row, "expected_intent", None) if has_expected else None

        nlu_result = nlu_classify(engine, utterance)
        time.sleep(args.delay)
        df_result  = df_fn(utterance, f"{args.session_prefix}-{i}")

        print_row(i, utterance, nlu_result, df_result, expected)

        rows.append({
            "utterance":      utterance,
            "expected":       expected,
            "nlu_intent":     nlu_result["intent"],
            "nlu_confidence": nlu_result["confidence"],
            "nlu_stage":      nlu_result["stage"],
            "df_intent":      df_result["intent"],
            "df_confidence":  df_result["confidence"],
            "nlu_correct":    (nlu_result["intent"] == expected) if expected else None,
            "df_correct":     (df_result["intent"]  == expected) if expected else None,
            "agree":          nlu_result["intent"] == df_result["intent"],
        })

    results = pd.DataFrame(rows)
    n = len(results)
    agree_count = results["agree"].sum()

    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  Agreement: {agree_count}/{n} ({100*agree_count/n:.1f}%)"
          f" -- both engines gave the same intent{RESET}")

    if has_expected:
        nlu_acc    = results["nlu_correct"].sum()
        df_acc     = results["df_correct"].sum()
        both_right = (results["nlu_correct"] & results["df_correct"]).sum()
        nlu_only   = (results["nlu_correct"] & ~results["df_correct"]).sum()
        df_only    = (~results["nlu_correct"] & results["df_correct"]).sum()
        both_wrong = (~results["nlu_correct"] & ~results["df_correct"]).sum()

        print(f"\n{BOLD}  Accuracy vs expected:{RESET}")
        print(f"    On-device NLU : {GREEN}{nlu_acc}/{n} ({100*nlu_acc/n:.1f}%){RESET}")
        print(f"    Dialogflow    : {GREEN}{df_acc}/{n}  ({100*df_acc/n:.1f}%){RESET}")
        print(f"\n{BOLD}  Breakdown:{RESET}")
        print(f"    Both correct  : {both_right}")
        print(f"    NLU only      : {nlu_only}  (on-device wins)")
        print(f"    DF only       : {df_only}   (Dialogflow wins)")
        print(f"    Both wrong    : {both_wrong}")

        print(f"\n{BOLD}  Per-intent breakdown:{RESET}")
        print(f"  {'Intent':<32} {'NLU':>6}  {'DF':>6}")
        print("  " + "-" * 50)
        for intent in sorted(results["expected"].dropna().unique()):
            sub   = results[results["expected"] == intent]
            nlu_ok = sub["nlu_correct"].sum()
            df_ok  = sub["df_correct"].sum()
            tot    = len(sub)
            nlu_c  = GREEN if nlu_ok == tot else (YELLOW if nlu_ok >= tot / 2 else RED)
            df_c   = GREEN if df_ok  == tot else (YELLOW if df_ok  >= tot / 2 else RED)
            print(f"  {intent:<32} {nlu_c}{nlu_ok}/{tot}{RESET}    {df_c}{df_ok}/{tot}{RESET}")

    print(f"{BOLD}{'='*72}{RESET}\n")

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
