"""
Loan decision assistant: EDUCATIONAL PROTOTYPE (Streamlit).

Flow: applicant details -> risk model + policy rules -> RAG explanation (AI DRAFT)
      -> automated checks -> loan officer checkpoint -> message released + audit log.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud, secret GEMINI_API_KEY in the app's Secrets settings.
"""
import datetime as dt
import os
import re

import pandas as pd
import streamlit as st

import rag
from decision_engine import DEMO

st.set_page_config(page_title="Loan decision assistant (prototype)", page_icon="🏦", layout="wide")

DISCLAIMER = ("Educational prototype. Not financial, medical or legal advice. "
              "All applicant data is synthetic. Do not enter real personal data.")
CUSTOM = dict(age=35, employment_type="Salaried", years_in_current_job=5.0, monthly_income=60000,
              existing_monthly_emi=0, credit_score=700, credit_history_years=6.0,
              num_past_delinquencies=0, loan_amount=300000, tenure_months=36)
EMP = ["Salaried", "Self-employed", "Business owner"]


def inr(x):
    s = str(int(round(x)))
    if len(s) > 3:
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", s[:-3]) + "," + s[-3:]
    return "₹" + s


# ------------------------------------------------------------------ key and LLM
try:
    _key = st.secrets["GEMINI_API_KEY"]
except Exception:
    _key = os.environ.get("GEMINI_API_KEY")
if _key:
    os.environ["GEMINI_API_KEY"] = _key
HAVE_KEY = bool(os.environ.get("GEMINI_API_KEY"))

for k, v in {"log": [], "case": None, "decision": None, "last_choice": None, "llm_source": ""}.items():
    st.session_state.setdefault(k, v)


def llm_with_fallback(prompt, res):
    """Real LLM when possible. If it fails (quota, outage) fall back to a template and say so."""
    if st.session_state.get("template_mode") or not HAVE_KEY:
        st.session_state.llm_source = "template (no LLM call)"
        return rag.mock_llm(prompt, res)
    try:
        text = rag.call_llm(prompt, res)
        st.session_state.llm_source = f"LLM ({rag.MODEL_NAME})"
        return text
    except Exception as e:
        st.session_state.llm_source = f"template (LLM unavailable: {str(e)[:80]})"
        return rag.mock_llm(prompt, res)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🏦 Loan assistant")
st.sidebar.warning(DISCLAIMER)
choice = st.sidebar.selectbox("Load a demo applicant", ["Custom"] + list(DEMO), key="choice",
                              format_func=lambda x: x.replace("_", " ").capitalize())
st.sidebar.checkbox("Template mode (no LLM call)", value=not HAVE_KEY, key="template_mode",
                    help="Use this if the LLM quota is exhausted. The explanation is then a fixed template.")
if not HAVE_KEY:
    st.sidebar.info("No GEMINI_API_KEY found, so template explanations are used.")
if st.session_state.last_choice != choice:
    st.session_state.last_choice, st.session_state.case, st.session_state.decision = choice, None, None

d = {**CUSTOM, **(DEMO[choice] if choice != "Custom" else {})}
tab1, tab2, tab3 = st.tabs(["Assess an application", "Decision log (audit trail)", "About and limits"])

# ------------------------------------------------------------------ tab 1: form
with tab1:
    st.title("Loan decision and explanation")
    st.caption(DISCLAIMER)
    with st.form("applicant_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 80, int(d["age"]), key=f"{choice}_age")
        emp = c1.selectbox("Employment type", EMP, EMP.index(d["employment_type"]), key=f"{choice}_emp")
        yrs = c1.number_input("Years in current job or business", 0.0, 40.0, float(d["years_in_current_job"]), 0.5, key=f"{choice}_yrs")
        inc = c2.number_input("Monthly income (₹)", 0, 2_000_000, int(d["monthly_income"]), 1000, key=f"{choice}_inc")
        emi0 = c2.number_input("Existing monthly EMI (₹)", 0, 1_000_000, int(d["existing_monthly_emi"]), 500, key=f"{choice}_emi")
        amt = c2.number_input("Loan amount requested (₹)", 10_000, 5_000_000, int(d["loan_amount"]), 10_000, key=f"{choice}_amt")
        ten_opts = [12, 24, 36, 48, 60]
        ten = c2.selectbox("Tenure (months)", ten_opts, ten_opts.index(int(d["tenure_months"])), key=f"{choice}_ten")
        no_score = c3.checkbox("No credit score on file", d["credit_score"] is None, key=f"{choice}_noscore")
        score = c3.number_input("Credit score (300 to 900)", 300, 900, int(d["credit_score"] or 700), key=f"{choice}_score")
        hist = c3.number_input("Credit history (years)", 0.0, 40.0, float(d["credit_history_years"]), 0.5, key=f"{choice}_hist")
        delq = c3.number_input("Past delinquencies", 0, 10, int(d["num_past_delinquencies"]), key=f"{choice}_delq")
        submitted = st.form_submit_button("Assess application", type="primary")

    if submitted:
        a = dict(age=int(age), employment_type=emp, years_in_current_job=float(yrs), monthly_income=float(inc),
                 existing_monthly_emi=float(emi0), credit_score=None if no_score else int(score),
                 credit_history_years=float(hist), num_past_delinquencies=int(delq),
                 loan_amount=float(amt), tenure_months=int(ten))
        try:
            with st.spinner("Scoring and writing the draft explanation..."):
                out = rag.explain(a, llm=llm_with_fallback)
        except Exception as e:
            st.error(f"Could not assess this application: {e}")
            out = None
        if out:
            cid = "APP-" + dt.datetime.now().strftime("%H%M%S")
            st.session_state.case = {"id": cid, "a": a, "out": out, "source": st.session_state.llm_source, "norag": None}
            st.session_state.decision = None
            if out["result"]["recommendation"] == "APPROVE":
                st.session_state.log.append({
                    "time": dt.datetime.now().strftime("%H:%M:%S"), "applicant": cid,
                    "system_recommendation": "APPROVE", "risk_score": out["result"]["pd"],
                    "final_decision": "APPROVE (automatic, clause 6.1)", "officer": "-", "override": False,
                    "reason": "", "message_edited": False})

    # -------------------------------------------------------------- results
    case = st.session_state.case
    if case:
        out, res, a = case["out"], case["out"]["result"], case["a"]
        rec = res["recommendation"]
        st.divider()
        st.markdown(f"### System recommendation: :{ {'APPROVE': 'green', 'REVIEW': 'orange', 'REJECT': 'red'}[rec] }[{rec}]")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model risk score (PD)", f"{res['pd']:.3f}", help="Approve below 0.10, review 0.10 to 0.25, reject at 0.25 or above")
        m2.metric("Debt-to-income", f"{res['dti']:.2f}", help="Policy limit 0.40 (0.50 with review)")
        m3.metric("Loan-to-income", f"{res['loan_to_income']:.2f}", help="Policy limit 2.0")
        m4.metric("Proposed EMI", inr(res["features"]["proposed_emi"]))

        rows = []
        for key, label in (("reject_reasons", "Not eligible"), ("review_reasons", "Needs review"), ("not_covered", "Not covered by policy")):
            if key in rag._keys(res):
                rows += [{"Type": label, "Clause": x["clause"], "Reason": x["reason"]} for x in res[key]]
        with st.expander("Rules that fired", expanded=bool(rows)):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch") if rows else st.write("No rule fired.")

        needs_human = res["human_review_required"]
        st.markdown("#### Explanation: " + ("AI DRAFT for the loan officer" if needs_human else "message to applicant"))
        st.info(out["explanation"])
        st.caption(f"Written by: {case['source']}. Clauses given to the writer: {', '.join(out['clause_ids'])}")

        c = out["checks"]
        if c["passed"]:
            st.success("Automated checks passed: only real clauses cited, every fired rule cited, numbers match the policy and applicant, no protected terms.")
        else:
            labels = {"invented_clauses": "cites clauses that do not exist", "clauses_not_provided": "cites clauses it was not given",
                      "fired_clauses_missing": "skips rules that applied", "unsupported_numbers": "uses numbers not in the policy or facts",
                      "protected_terms": "mentions a protected attribute", "changed_decision": "contradicts the system decision"}
            issues = [f"{labels[k]}: {c[k]}" if isinstance(c[k], list) else labels[k] for k in labels if c[k]]
            if not c["says_not_covered_when_needed"]:
                issues.append("does not say the case is not covered by the policy")
            st.error("Automated checks FAILED. Edit the message before releasing it.\n\n- " + "\n- ".join(issues))
        if c.get("verbatim_copy"):
            st.warning(f"Style warning: wording copied word for word from clause(s) {', '.join(c['verbatim_copy'])}.")

        with st.expander("Policy clauses the writer was given"):
            for i in out["clause_ids"]:
                cl = rag._cache["clauses"][i]
                st.markdown(f"**{i}** ({cl['section']}): {cl['text']}")

        with st.expander("Why RAG matters: same LLM WITHOUT the policy"):
            if st.session_state.get("template_mode") or not HAVE_KEY:
                st.write("Turn off template mode and add an API key to run this comparison.")
            elif st.button("Generate an ungrounded explanation", key="compare"):
                case["norag"] = rag.explain(a, llm=llm_with_fallback, use_rag=False)["explanation"]
            if case["norag"]:
                st.warning("Not grounded in the policy. Look for invented figures, promises and timelines.")
                st.write(case["norag"])

        # ---------------------------------------------------------- human checkpoint
        st.divider()
        if not needs_human:
            st.success("Approved under clause 6.1. Approvals do not need human review, so the message above is released.")
        else:
            st.markdown("### 🧑‍⚖️ Human checkpoint: loan officer decision required")
            why = {"REJECT": "Every rejection must be confirmed by a loan officer before it is communicated (clauses 6.6, 9.1).",
                   "REVIEW": "This case needs manual review (clauses 6.3, 9.2" + (", 9.4)." if res["not_covered"] else ").")}[rec]
            st.write(why + " Nothing is sent to the applicant until a decision is recorded.")
            dec = st.session_state.decision
            if dec:
                st.success(f"Decision recorded: **{dec['final']}** by {dec['officer']}. Message released to the applicant:")
                st.info(dec["message"])
            else:
                options = ["Confirm rejection", "Override: approve"] if rec == "REJECT" else ["Approve after review", "Reject after review"]
                with st.form(f"officer_form_{case['id']}"):
                    officer = st.text_input("Your name or ID")
                    choice_o = st.radio("Decision", options)
                    reason = st.text_area("Reason (required; it goes in the audit log)")
                    senior = st.checkbox("Senior Credit Manager has approved this override (clause 7.2)")
                    msg = st.text_area("Message to applicant (edit the AI draft if your decision differs from it)", value=out["explanation"], height=160)
                    fixed = True if c["passed"] else st.checkbox("I have corrected the problems flagged by the automated checks")
                    go = st.form_submit_button("Record decision and release message", type="primary")
                if go:
                    approve_override = choice_o == "Override: approve"
                    reject_clauses = {x["clause"] for x in res["reject_reasons"]}
                    errors = []
                    if not officer.strip(): errors.append("Enter your name or ID.")
                    if not reason.strip(): errors.append("A written reason is required (clause 7.1).")
                    if approve_override and reject_clauses & {"3.5", "4.4"}:
                        errors.append("This override is not allowed: credit score below 600 or debt-to-income above 0.50 (clause 7.3).")
                    elif approve_override and (reject_clauses - {"5.4"}) and not senior:
                        errors.append("Overriding a 'not eligible' rule needs Senior Credit Manager approval (clause 7.2).")
                    if not fixed: errors.append("Confirm that you corrected the flagged problems.")
                    if errors:
                        st.error("\n\n".join(errors))
                    else:
                        final = {"Confirm rejection": "REJECT", "Override: approve": "APPROVE (override)",
                                 "Approve after review": "APPROVE", "Reject after review": "REJECT"}[choice_o]
                        st.session_state.decision = {"final": final, "officer": officer.strip(), "message": msg}
                        st.session_state.log.append({
                            "time": dt.datetime.now().strftime("%H:%M:%S"), "applicant": case["id"],
                            "system_recommendation": rec, "risk_score": res["pd"], "final_decision": final,
                            "officer": officer.strip(), "override": approve_override, "reason": reason.strip(),
                            "message_edited": msg.strip() != out["explanation"].strip()})
                        st.rerun()

# ------------------------------------------------------------------ tab 2: audit log
with tab2:
    st.subheader("Decision log")
    st.caption("Kept in this browser session only and cleared when the app restarts. A real bank would write this to an audit database.")
    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log)
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("Download log as CSV", df.to_csv(index=False), "decision_log.csv", "text/csv")
        st.caption(f"Overrides: {int(df['override'].sum())} of {len(df)} decisions. The Credit Risk Committee reviews every override each quarter (clause 7.4).")
    else:
        st.write("No decisions yet.")

# ------------------------------------------------------------------ tab 3: about
with tab3:
    st.subheader("How it works")
    st.markdown("""
1. A **logistic regression risk model** estimates the probability of default.
2. **Policy rules** (the bank's lending policy) decide which clauses apply and set the recommendation: approve, review or reject.
3. A **RAG component** retrieves those clauses and an LLM writes a plain-language draft that cites them. The LLM never makes or changes the decision.
4. **Automated checks** test the draft for invented clauses, unsupported numbers, protected terms and a changed decision.
5. A **loan officer** confirms every rejection and decides every review case before anything is released.
""")
    st.subheader("Known limits")
    st.markdown("""
- **Synthetic data.** The 8,000 applicants were generated from assumptions we chose. Reported accuracy says nothing about real borrowers.
- **Threshold depends on assumed costs.** A missed default costs 50% of the loan and a wrongly rejected good customer costs 8% of the loan. Different costs give a different threshold.
- **Many good applicants are rejected.** At the cost-based threshold the model rejects about 29% of good applicants. The human checkpoint exists to catch these.
- **Checks are not perfect.** They cannot tell whether a cited clause is truly relevant, and they have raised false alarms.
- **Only known policy gaps are detected automatically.** Unknown gaps depend on the LLM saying "not covered".
- **Approvals are not reviewed.** A real bank would sample auto-approvals for audit.
- **Privacy.** Real applicant data must not go to a public LLM API. This prototype uses synthetic data only.
""")
    st.caption(DISCLAIMER)
