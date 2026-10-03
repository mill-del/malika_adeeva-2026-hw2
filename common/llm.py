"""Shared helpers for HW2: model call, JSON parsing, schema validation, tables."""
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from jsonschema.validators import validator_for
from openai import OpenAI

load_dotenv()

MODEL = "gpt-5.6-luna"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------- data loading (always UTF-8: Windows defaults to cp1251) ----------

def load_json(name):
    with open(DATA / name, encoding="utf-8") as f:
        return json.load(f)


def load_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------- model call ----------

def call(messages, json_mode=True):
    """Send messages, return (text, usage).

    json_mode=True uses response_format json_object (NOT strict schema),
    so 'did it parse' and 'did it validate' stay meaningful checks.
    Note: json_object mode requires the word 'JSON' somewhere in messages.
    """
    kwargs = {"model": MODEL, "messages": messages}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client().chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    return text, resp.usage


# ---------- parsing and validation ----------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def parse(text):
    """Return (obj, None) on success, (None, error_message) on failure."""
    cleaned = _FENCE.sub("", text or "").strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"
    if not isinstance(obj, dict):
        return None, f"top-level value is {type(obj).__name__}, not object"
    return obj, None


def validate(obj, schema):
    """Return (True, None) if obj matches schema, else (False, first_error)."""
    cls = validator_for(schema)
    errors = sorted(cls(schema).iter_errors(obj), key=lambda e: list(e.path))
    if not errors:
        return True, None
    e = errors[0]
    where = "/".join(str(p) for p in e.path) or "(root)"
    return False, f"{where}: {e.message}"


# ---------- the answer contract used in every sublab ----------

DECISIONS = ["granted", "refused", "more_info", "not_found"]

ANSWER_SCHEMA = {
    "type": "object",
    "required": ["applicant_id", "found", "decision", "amount",
                 "missing_documents", "reason"],
    "additionalProperties": False,
    "properties": {
        "applicant_id": {"type": ["string", "null"]},
        "found": {"type": "boolean"},
        "decision": {"enum": DECISIONS},
        "amount": {"type": "integer", "minimum": 0},
        "missing_documents": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
}

CHECKED_FIELDS = ["found", "decision", "amount", "missing_documents"]


def same_field(field, a, b):
    """Compare one structured field; documents compared as sets."""
    if field == "missing_documents":
        return set(a or []) == set(b or [])
    return a == b


# ---------- markdown table for pasting into SUBMISSION.md ----------

def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)
