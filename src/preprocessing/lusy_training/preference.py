from __future__ import annotations
import hashlib, orjson

def prompt_fingerprint(prompt: list[dict]) -> str:
    canonical=[{'role':m.get('role'),'content':m.get('content')} for m in prompt]
    return hashlib.sha256(orjson.dumps(canonical,option=orjson.OPT_SORT_KEYS)).hexdigest()

def make_pair(prompt, rejected, chosen, *, chosen_format_valid: bool, chosen_identity_confidence: float, kind: str, explicit: bool, metadata: dict | None=None):
    if not chosen_format_valid or chosen_identity_confidence>=.60 or not str(chosen).strip() or str(chosen).strip()==str(rejected).strip(): return None
    row={'prompt':prompt,'chosen':chosen,'rejected':rejected,'prompt_fingerprint':prompt_fingerprint(prompt),'pair_type':kind,'pair_strength':'strict' if explicit else 'weak'}
    if metadata: row.update(metadata)
    return row
