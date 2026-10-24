"""
RAG component (EDUCATIONAL PROTOTYPE): policy clauses -> grounded explanation -> automatic checks.

Pipeline:  decision_engine.evaluate()  ->  choose clauses  ->  prompt  ->  LLM  ->  verify_explanation()
The LLM only EXPLAINS the decision. It never makes or changes it.

    python rag.py --mock      # no API key: shows retrieval, prompt and checks with a template "LLM"
    python rag.py             # real LLM (needs GEMINI_API_KEY in the environment)
    python rag.py --no-rag    # same LLM WITHOUT the policy, to show why RAG matters
"""
import argparse
import os
import re
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from decision_engine import DEMO, ROOT, evaluate

POLICY_PATH = ROOT / "policy" / "lending_policy.md"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")   # use a model your key can access

# ---------------------------------------------------------------- policy -> clauses
def load_clauses(path=POLICY_PATH):
    clauses, section = {}, ""
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        m = re.match(r"\*\*(\d+\.\d+)\*\*\s+(.*)", line)
        if m:
            clauses[m.group(1)] = {"id": m.group(1), "section": section, "text": m.group(2)}
    return clauses


class Retriever:
    """Keyword (TF-IDF) retrieval over clauses. Fine for ~65 short clauses; embeddings are an upgrade."""
    def __init__(self, clauses):
        self.ids = list(clauses)
        docs = [f"{clauses[i]['section']} {clauses[i]['text']}" for i in self.ids]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.mat = self.vec.fit_transform(docs)

    def top(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        return [self.ids[i] for i in sims.argsort()[::-1][:k] if sims[i] > 0]


def _keys(res):
    """Which reason lists matter for this decision (a rejection is explained by rejection reasons only)."""
    return ("reject_reasons", "not_covered") if res["recommendation"] == "REJECT" \
        else ("reject_reasons", "review_reasons", "not_covered")


def fired_clauses(res):
    """Clauses the decision engine says apply, in order, without repeats."""
    out = []
    for key in _keys(res):
        for x in res[key]:
            if x["clause"] not in out:
                out.append(x["clause"])
    return out


def choose_clauses(res, clauses, retriever, k=3):
    """Hybrid: clauses that fired (deterministic) + decision rules + top-k retrieved."""
    forced = fired_clauses(res)
    rec = res["recommendation"]
    forced += {"APPROVE": ["6.1", "5.2"],
               "REJECT": ["6.2", "6.6", "11.3", "11.4"],
               "REVIEW": ["6.3", "9.2"]}[rec]
    if res["not_covered"]:
        forced += ["10.1", "10.2", "9.4"]
    query = " ".join(x["reason"] for key in _keys(res) for x in res[key])
    extra = retriever.top(query or "approval conditions eligibility", k)
    ids = []
    for i in forced + extra:
        if i in clauses and i not in ids:
            ids.append(i)
    return ids, extra


# ---------------------------------------------------------------- prompt
SYSTEM_RULES = """You write a DRAFT loan decision explanation for a bank. A loan officer reads it before anything reaches the applicant.
Rules:
1. Use ONLY the policy clauses listed below. Never use outside lending knowledge, other banks' rules, timelines, or numbers that are not in the clauses or applicant facts.
2. Cite each rule you rely on as [Clause 4.4], placed right after the statement it supports. Cite only clause numbers from the list below, and only a clause that truly supports that statement. Never cite a section number alone.
3. The decision was already made by the bank's system. Explain it. Never change it, soften it, or promise a different outcome or a date.
4. If the situation is marked NOT COVERED, say plainly that the policy does not cover it and that a loan officer will decide. Do not guess a rule.
5. Never mention or hint at gender, religion, caste, marital status or city. Do not mention fairness rules at all.
6. Speak directly to the applicant ("your income", "your credit score"). Plain language, under 120 words, no jargon: say "risk score", never "PD".
7. Explain only the reasons that apply to THIS applicant, in your own words. Never copy a clause sentence word for word, and never recite the bank's internal procedures (what the system or a loan officer "must" do in every case).
8. For REVIEW or REJECT, add one short sentence saying a loan officer looks at the case before anything is final. For REJECT, mention the 30-day review and 90-day reapply options only if those clauses are listed."""


def applicant_facts(a, res):
    f = res["features"]
    return (f"Age {a['age']}; employment {a['employment_type']}; years in job {a.get('years_in_current_job')}; "
            f"monthly income Rs {a.get('monthly_income')}; credit score {a.get('credit_score')}; "
            f"credit history {a['credit_history_years']} years; past delinquencies {a['num_past_delinquencies']}; "
            f"loan Rs {a['loan_amount']} for {a['tenure_months']} months; proposed EMI Rs {int(f['proposed_emi'])}; "
            f"debt-to-income {res['dti']}; loan-to-income {res['loan_to_income']}; model risk score {res['pd']}.")


def build_prompt(a, res, clauses, ids):
    ctx = "\n".join(f"[Clause {i}] ({clauses[i]['section']}) {clauses[i]['text']}" for i in ids)
    flags = []
    for key, label in [t for t in (("reject_reasons", "NOT ELIGIBLE"), ("review_reasons", "NEEDS REVIEW"), ("not_covered", "NOT COVERED")) if t[0] in _keys(res)]:
        flags += [f"- {label}: clause {x['clause']} - {x['reason']}" for x in res[key]]
    return (f"{SYSTEM_RULES}\n\nPOLICY CLAUSES:\n{ctx}\n\nAPPLICANT FACTS:\n{applicant_facts(a, res)}\n\n"
            f"SYSTEM DECISION: {res['recommendation']}\nRULES THAT FIRED:\n" + ("\n".join(flags) or "- none") +
            "\n\nWrite the explanation now.")


def build_prompt_no_rag(a, res):
    """Comparison only: the same task but WITHOUT the policy."""
    return ("You write loan decision explanations for a bank. Explain this decision to the applicant in under 150 words.\n"
            f"APPLICANT FACTS: {applicant_facts(a, res)}\nSYSTEM DECISION: {res['recommendation']}")


# ---------------------------------------------------------------- LLM
def call_llm(prompt, res=None, retries=3):
    """Gemini via the google-genai SDK. Swap this one function to change provider."""
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for n in range(retries):
        try:
            r = client.models.generate_content(model=MODEL_NAME, contents=prompt,
                                               config={"temperature": 0.2})
            return r.text.strip()
        except Exception as e:                       # rate limits on the free tier are common
            if n == retries - 1:
                raise
            time.sleep(5 * (n + 1))


def mock_llm(prompt, res):
    """Template stand-in so the pipeline can be tested without a key. NOT part of the final evaluation."""
    cites = " ".join(f"[Clause {c}]" for c in fired_clauses(res))
    if res["not_covered"]:
        return f"Your application raises a situation this policy does not cover {cites}. A loan officer will decide."
    if res["recommendation"] == "APPROVE":
        return "Your application meets the policy requirements [Clause 6.1] and has a low risk score [Clause 5.2]."
    why = "; ".join(x["reason"] for k in ("reject_reasons", "review_reasons") for x in res[k])
    return f"We could not approve this automatically because {why} {cites}. A loan officer will review it."


# ---------------------------------------------------------------- automatic checks
BANNED = r"\b(gender|male|female|woman|women|man|men|religion|caste|married|marital|city)\b"


def _floats(strings):
    out = []
    for n in strings:
        try:
            out.append(float(n.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def _words(t):
    return re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()


def verbatim_copies(text, clauses, ids, n=10):
    """Clauses whose wording (n+ consecutive words) was copied straight into the explanation."""
    blob = " ".join(_words(text))
    hits = []
    for i in ids:
        cw = _words(clauses[i]["text"])
        if any(" ".join(cw[j:j + n]) in blob for j in range(len(cw) - n + 1)):
            hits.append(i)
    return hits


def verify_explanation(text, res, clauses, provided_ids, a):
    cited = set(re.findall(r"[Cc]lause\s+(\d+(?:\.\d+)?)", text))
    required = set(fired_clauses(res))
    context_text = " ".join(clauses[i]["text"] for i in provided_ids) + " " + applicant_facts(a, res)
    allowed = _floats(re.findall(r"\d[\d,\.]*", context_text))
    used = _floats(re.findall(r"\d[\d,\.]*", re.sub(r"[Cc]lause\s+\d+(?:\.\d+)?", "", text)))
    # a number is supported if it matches a number in the policy/facts (allowing for rounding)
    unsupported = sorted({u for u in used if not any(abs(u - m) <= 0.006 + 0.0005 * abs(m) for m in allowed)})
    rec = res["recommendation"]
    wrong_outcome = (rec != "APPROVE" and re.search(r"\b(congratulations|(is|been|are|was|will be) approved|pleased to approve)\b", text, re.I)) \
        or (rec == "APPROVE" and re.search(r"\b(rejected|declined|not eligible)\b", text, re.I))
    checks = {
        "invented_clauses": sorted(cited - set(clauses)),                 # cites a clause that does not exist
        "clauses_not_provided": sorted((cited & set(clauses)) - set(provided_ids)),
        "fired_clauses_missing": sorted(required - cited),                # skipped a rule that applied
        "unsupported_numbers": unsupported,                               # number not found in policy or facts
        "protected_terms": re.findall(BANNED, text, flags=re.I),
        "changed_decision": bool(wrong_outcome),
        "verbatim_copy": verbatim_copies(text, clauses, provided_ids),   # style warning only
        "says_not_covered_when_needed": (not res["not_covered"]) or bool(re.search(r"not cover", text, re.I)),
    }
    checks["passed"] = not (checks["invented_clauses"] or checks["clauses_not_provided"] or checks["fired_clauses_missing"]
                            or checks["unsupported_numbers"] or checks["protected_terms"] or checks["changed_decision"]
                            or not checks["says_not_covered_when_needed"])
    return checks


# ---------------------------------------------------------------- one call for the app
_cache = {}


def explain(a, llm=call_llm, use_rag=True):
    if "clauses" not in _cache:
        _cache["clauses"] = load_clauses()
        _cache["retriever"] = Retriever(_cache["clauses"])
    clauses, retriever = _cache["clauses"], _cache["retriever"]
    res = evaluate(a)
    if use_rag:
        ids, extra = choose_clauses(res, clauses, retriever)
        prompt = build_prompt(a, res, clauses, ids)
    else:
        ids, extra, prompt = [], [], build_prompt_no_rag(a, res)
    text = llm(prompt, res)
    checks = verify_explanation(text, res, clauses, ids, a) if use_rag else None
    return {"result": res, "clause_ids": ids, "retrieved_extra": extra, "prompt": prompt,
            "explanation": text, "checks": checks}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()
    llm = mock_llm if args.mock else call_llm
    for name, a in DEMO.items():
        out = explain(a, llm=llm, use_rag=not args.no_rag)
        print(f"\n===== {name}: {out['result']['recommendation']} (PD {out['result']['pd']})")
        print("clauses given to LLM:", out["clause_ids"], "| extra from retrieval:", out["retrieved_extra"])
        if args.show_prompt:
            print(out["prompt"])
        print("EXPLANATION:", out["explanation"])
        if out["checks"]:
            print("CHECKS:", {k: v for k, v in out["checks"].items() if v not in ([], False) or k == "passed"})
