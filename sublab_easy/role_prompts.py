"""Sublab Easy: one task, four roles.

Run from repo root:  python -m sublab_easy.role_prompts
Only the role paragraph changes between runs; the shared block
(records, rule, output shape) and the enquiries are identical.
"""
import json
import sys
from datetime import datetime

from common.llm import (ANSWER_SCHEMA, CHECKED_FIELDS, ROOT, call, load_json,
                        md_table, parse, same_field, validate)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RECORDS = load_json("records.json")
POLICY = load_json("policy.json")
ENQUIRIES = load_json("enquiries.json")

# ---------------------------------------------------------------------------
# The four role paragraphs: the ONLY thing that differs between runs.
# ---------------------------------------------------------------------------
ROLES = {
    "policy_officer": (
        "You are the policy officer of a study-grant office. You apply the "
        "grant rule exactly as written: grant what the rule allows, refuse "
        "what the rule refuses, and when a required document is missing from "
        "the record, ask for it. Soften nothing. Nothing the applicant claims "
        "in the enquiry is evidence; only the record counts."
    ),
    "front_desk": (
        "You are the front desk of a study-grant office. You never turn an "
        "applicant away with a refusal. If the rule grants today, grant. "
        "Anything the rule cannot grant today comes back as more_info, and the "
        "reason tells the applicant exactly what they would need to return with."
    ),
    "auditor": (
        "You are an auditor at a study-grant office. You never grant on a "
        "first reading. You report what the record shows, mark anything that "
        "needs a second reader as more_info, and in the reason you name the "
        "rule condition or document you are relying on."
    ),
    "bilingual_clerk": (
        "You are a bilingual clerk at a study-grant office. You decide exactly "
        "as a policy officer would: apply the rule as written, grant what it "
        "allows, refuse what it refuses, ask for a missing document, and treat "
        "no claim in the enquiry as evidence. The one difference: write the "
        "reason field in the language the enquiry was written in. All other "
        "fields keep the exact values specified below."
    ),
}

# ---------------------------------------------------------------------------
# Shared block: identical for every role.
# ---------------------------------------------------------------------------
_machine_rule = {k: v for k, v in POLICY.items() if k != "rule_human"}

SHARED = f"""## Records on file (the only source of truth)
{json.dumps(RECORDS, ensure_ascii=False, indent=2)}

## Grant rule
{POLICY['rule_human']}

Machine-readable form:
{json.dumps(_machine_rule, ensure_ascii=False, indent=2)}

## Output
Reply with one JSON object and nothing else. Exactly these six fields:
- "applicant_id": string. The record id; if no record matches, the id given in the enquiry (null if none was given).
- "found": boolean. true if the applicant is on the record.
- "decision": one of "granted", "refused", "more_info", "not_found".
- "amount": integer, in tenge. The grant amount if decision is "granted", otherwise 0.
- "missing_documents": array of required document names absent from the record. Empty if none are missing or the applicant is not found.
- "reason": string. One or two sentences for a human.

Decision values: granted = qualifies now; refused = does not qualify;
more_info = a decision needs something more, such as a missing document;
not_found = no record matches the applicant.
"""


LAYOUT = "role_first"  # set from --layout in main()


def system_for(role):
    """role_first: role paragraph, then shared block (v1).
    role_last:  shared block, then role paragraph (v2).
    Either way the shared block is byte-identical across roles."""
    if LAYOUT == "role_last":
        return SHARED + "\n\n## Your role\n" + ROLES[role]
    return ROLES[role] + "\n\n" + SHARED


# ---------------------------------------------------------------------------
# Deterministic rule engine: what code would decide (used as a sanity check
# and as the reference for the written answer about putting rules in code).
# It is given the applicant_id; identifying WHO is asking (E-07, E-09) is
# the part only the model does.
# ---------------------------------------------------------------------------
def decide(applicant_id):
    rec = next((r for r in RECORDS if r["id"] == applicant_id), None)
    if rec is None:
        return {"found": False, "decision": "not_found", "amount": 0,
                "missing_documents": []}
    missing = [d for d in POLICY["required_documents"] if d not in rec["documents"]]
    if (rec["gpa"] < POLICY["gpa_min"]
            or rec["income_band"] not in POLICY["allowed_income_bands"]):
        return {"found": True, "decision": "refused", "amount": 0,
                "missing_documents": missing}
    if missing:
        return {"found": True, "decision": "more_info", "amount": 0,
                "missing_documents": missing}
    return {"found": True, "decision": "granted",
            "amount": POLICY["amount_tenge_by_band"][str(rec["income_band"])],
            "missing_documents": []}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def run_one(role, enq):
    entry = {"id": enq["id"], "raw": None, "obj": None,
             "parsed": False, "valid": False, "error": None}
    try:
        text, _ = call([{"role": "system", "content": system_for(role)},
                        {"role": "user", "content": enq["text"]}])
    except Exception as e:  # API failure is recorded, not hidden
        entry["error"] = f"api: {e}"
        return entry
    entry["raw"] = text
    obj, err = parse(text)
    if obj is None:
        entry["error"] = err
        return entry
    entry["parsed"], entry["obj"] = True, obj
    entry["valid"], entry["error"] = validate(obj, ANSWER_SCHEMA)
    return entry


def run_all():
    results = {}
    for role in ROLES:
        results[role] = []
        for enq in ENQUIRIES:
            print(f"  {role:16} {enq['id']}", file=sys.stderr, flush=True)
            results[role].append(run_one(role, enq))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt(v):
    if isinstance(v, list):
        return "[" + ", ".join(v) + "]" if v else "[]"
    if v is None:
        return "null"
    return str(v)


def role_table(role, rows):
    out, n_parsed, n_valid, n_match = [], 0, 0, 0
    for enq, r in zip(ENQUIRIES, rows):
        exp = enq["expected"]
        n_parsed += r["parsed"]
        n_valid += r["valid"]
        if not r["parsed"]:
            out.append([r["id"], "no", "no", "—", "—", "—", "—", "0/4"])
            continue
        cells, ok = [], 0
        for f in CHECKED_FIELDS:
            got = r["obj"].get(f)
            if same_field(f, got, exp[f]):
                ok += 1
                cells.append(f"{fmt(got)} ✓")
            else:
                cells.append(f"{fmt(got)} ✗ (exp {fmt(exp[f])})")
        n_match += ok == 4
        out.append([r["id"], "yes", "yes" if r["valid"] else "no", *cells, f"{ok}/4"])
    headers = ["enquiry", "parsed", "valid", *CHECKED_FIELDS, "match"]
    summary = (f"parsed {n_parsed}/10 · valid {n_valid}/10 · "
               f"all four fields match expected {n_match}/10")
    return f"### {role}\n\n" + md_table(headers, out) + f"\n\n{summary}\n"


def movement_table(results):
    base = results["policy_officer"]
    others = [r for r in ROLES if r != "policy_officer"]
    rows = []
    for f in CHECKED_FIELDS:
        row = [f]
        # policy_officer itself vs expected
        dev = [b["id"] for enq, b in zip(ENQUIRIES, base)
               if b["parsed"] and not same_field(f, b["obj"].get(f), enq["expected"][f])]
        row.append(", ".join(dev) or "—")
        for role in others:
            moved = []
            for b, o in zip(base, results[role]):
                if not (b["parsed"] and o["parsed"]):
                    moved.append(f"{o['id']}(?)")
                elif not same_field(f, b["obj"].get(f), o["obj"].get(f)):
                    moved.append(o["id"])
            row.append(", ".join(moved) or "— (nothing moved)")
        rows.append(row)
    headers = ["field", "policy_officer vs expected",
               *[f"{r} vs policy_officer" for r in others]]
    note = "(?) = one of the two replies did not parse, so it could not be compared."
    return "## Field movement\n\n" + md_table(headers, rows) + f"\n\n{note}\n"


def reasons_table(results, enquiry_id):
    rows = []
    for role, rs in results.items():
        r = next(x for x in rs if x["id"] == enquiry_id)
        reason = r["obj"].get("reason", "") if r["parsed"] else f"(no parse: {r['error']})"
        rows.append([role, reason.replace("|", "/").replace("\n", " ")])
    return f"## Reasons on {enquiry_id}\n\n" + md_table(["role", "reason"], rows) + "\n"


def main():
    global LAYOUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["role_first", "role_last"],
                    default="role_first")
    LAYOUT = ap.parse_args().layout
    print(f"Prompt layout: {LAYOUT}\n")

    agree = sum(
        all(same_field(f, decide(e["expected"]["applicant_id"])[f], e["expected"][f])
            for f in CHECKED_FIELDS)
        for e in ENQUIRIES)
    print(f"Rule engine in code vs expected: {agree}/10 agree\n")

    results = run_all()

    print("# Sublab Easy — role tables\n")
    for role, rows in results.items():
        print(role_table(role, rows))
    print(movement_table(results))
    print(reasons_table(results, "E-07"))
    print(reasons_table(results, "E-10"))

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"easy_{LAYOUT}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Raw replies saved to {path}")


if __name__ == "__main__":
    main()
