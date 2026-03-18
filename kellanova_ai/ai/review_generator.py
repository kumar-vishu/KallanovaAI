"""
AI Business Review Generator
Supports two LLM providers:
  • Groq  — used automatically when GROQ_API_KEY is set (cloud deployment)
  • Ollama — local fallback for development (ollama serve)
All KPIs are pre-computed in Python — the LLM only writes the narrative.
"""
from __future__ import annotations
import re
import requests
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL


def _bold_numbers(text: str) -> str:
    """
    Post-process AI text so that currency amounts and percentages are
    rendered in bold Markdown regardless of what the LLM produced.
    Handles: $1,411  $23,340.38  $1.2M  $4.5K  12.5%  -10.4%
    Avoids double-bolding already-bolded spans.

    IMPORTANT: Streamlit's st.markdown() treats $...$ as LaTeX/KaTeX, which
    renders each character of the enclosed text as a separate math symbol.
    We escape every $ → \\$ so it renders as a literal dollar sign, not LaTeX.
    """
    # Strip any existing bold around numbers so we re-apply consistently
    # Handle both escaped (\$) and unescaped ($) variants
    text = re.sub(r'\*\*\\?(\$[\d,\.]+[KkMm]?)\*\*', r'\1', text)
    text = re.sub(r'\*\*(-?[\d,\.]+%)\*\*', r'\1', text)

    # Bold currency: $1,411 / $23,340.38 / $1.2M / $4.5K
    text = re.sub(
        r'(\$[\d,]+(?:\.\d+)?[KkMm]?)',
        r'**\1**',
        text,
    )
    # Bold percentages: -10.45% / +5.2% / 73%
    text = re.sub(
        r'([+\-]?\d+(?:\.\d+)?%)',
        r'**\1**',
        text,
    )
    # Escape ALL remaining $ signs followed by a digit so Streamlit's Markdown
    # parser does not interpret them as LaTeX inline-math delimiters ($...$).
    # \$ renders as a literal "$" in Markdown — both inside **bold** and in tables.
    text = re.sub(r'\$(\d)', r'\\$\1', text)
    return text


def _call_groq(prompt: str, max_tokens: int = 600) -> str:
    """Call the Groq cloud API and return the generated text."""
    try:
        from groq import Groq  # imported lazily — not needed on pure-local installs
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        raw = completion.choices[0].message.content.strip()
        return _bold_numbers(raw)
    except Exception as e:
        return _fallback_message(f"Groq error: {e}")


def _call_ollama(prompt: str, max_tokens: int = 600) -> str:
    """Call the local Ollama API and return the generated text."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.4},
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        return _bold_numbers(raw)
    except requests.exceptions.ConnectionError:
        return _fallback_message("Ollama not running. Start with: ollama serve")
    except Exception as e:
        return _fallback_message(str(e))


def _call_llm(prompt: str, max_tokens: int = 600) -> str:
    """
    Route to the correct LLM provider:
      • GROQ_API_KEY set  →  Groq  (cloud deployment)
      • otherwise         →  Ollama (local development)
    """
    if GROQ_API_KEY:
        return _call_groq(prompt, max_tokens)
    return _call_ollama(prompt, max_tokens)


def _fallback_message(reason: str) -> str:
    return f"⚠ AI review unavailable: {reason}"


# ── Territory Review ──────────────────────────────────────────────────────────
def generate_territory_review(territory_data: dict) -> str:
    prompt = f"""
You are a senior CPG retail analytics manager reviewing Kellanova NZ field sales performance.
Write a concise, professional business review (4–6 paragraphs) for the territory manager.

FORMATTING RULES — follow exactly:
- Output plain Markdown only. No HTML, no LaTeX.
- Do NOT wrap numbers in backticks. Never use `code` formatting for values.
- Write all dollar amounts and percentages inline in normal text.
- Use **bold** to emphasise key figures, store names, and action verbs.
- Each section heading must use ## (level 2).

TERRITORY DATA:
- Territory: {territory_data.get('territory_name')}
- Total Stores: {territory_data.get('total_stores')}
- Stores with Opportunities: {territory_data.get('stores_with_opportunity')}
- Total Opportunity Value: ${territory_data.get('total_opportunity_value', 0):,.0f}
- Hidden Opportunity (untapped demand): ${territory_data.get('total_hidden_opp', 0):,.0f}
- Promotion Compliance: {territory_data.get('promo_compliance_pct', 0):.1f}%
- Stores near Events: {territory_data.get('event_opp_stores')}
- Top Issue Types: {territory_data.get('top_issues', 'N/A')}
- Top Opportunity Stores: {territory_data.get('top_stores', 'N/A')}

Write the review with these sections:
## Performance Summary
## Key Opportunities and Root Causes
## Event-Driven Opportunities
## Recommended Priority Actions
## Outlook

Be specific, use the numbers provided, and focus on practical retail execution actions.
"""
    return _call_llm(prompt)


# ── Rep Review ────────────────────────────────────────────────────────────────
def generate_rep_review(rep_data: dict) -> str:
    prompt = f"""
You are a CPG field sales manager reviewing daily priorities for a Kellanova NZ sales rep.
Write a concise, motivating daily briefing (3–5 paragraphs) for the rep.

FORMATTING RULES — follow exactly:
- Output plain Markdown only. No HTML, no LaTeX.
- Do NOT wrap numbers in backticks. Never use `code` formatting for values.
- Write all dollar amounts and percentages inline in normal text.
- Use **bold** to emphasise key dollar figures, store names, and action verbs.
- Each section heading must use ## (level 2).

REP DATA:
- Rep Name: {rep_data.get('rep_name')}
- Territory: {rep_data.get('territory_name')}
- Stores Managed: {rep_data.get('stores_managed')}
- High Priority Stores Today: {rep_data.get('high_priority_stores')}
- Total Opportunity Value: ${rep_data.get('total_opportunity_value', 0):,.0f}
- Top Priority Store: {rep_data.get('top_store_name')} (${rep_data.get('top_store_opp', 0):,.0f})
- Key Issues: {rep_data.get('key_issues', 'N/A')}
- Events This Week: {rep_data.get('events_this_week', 'None')}
- Suggested Visit Order: {rep_data.get('visit_route', 'N/A')}

Write the briefing with these sections:
## Today's Opportunity Summary
## Priority Stores and Actions
## Event Opportunities
## Recommended Tasks for Today

Be direct and practical — this is a daily action briefing for a field rep.
"""
    return _call_llm(prompt)


# ── Store Review ──────────────────────────────────────────────────────────────
def generate_store_review(store_data: dict) -> str:
    # ── Period figures ────────────────────────────────────────────────────────
    l4w_actual   = store_data.get("l4w_actual",   store_data.get("actual_revenue", 0) * 4)
    l4w_expected = store_data.get("l4w_expected", store_data.get("expected_revenue", 0) * 4)
    l4w_gap      = l4w_actual - l4w_expected

    l13w_actual   = store_data.get("l13w_actual",   l4w_actual * 3.25)
    l13w_expected = store_data.get("l13w_expected", l4w_expected * 3.25)
    l13w_gap      = l13w_actual - l13w_expected

    ytd_actual   = store_data.get("ytd_actual",   l4w_actual * 12)
    ytd_expected = store_data.get("ytd_expected", l4w_expected * 12)
    ytd_gap      = ytd_actual - ytd_expected

    l4w_pct  = (l4w_gap  / l4w_expected  * 100) if l4w_expected  else 0
    l13w_pct = (l13w_gap / l13w_expected * 100) if l13w_expected else 0
    ytd_pct  = (ytd_gap  / ytd_expected  * 100) if ytd_expected  else 0

    # ── Category table ────────────────────────────────────────────────────────
    cat_rows = store_data.get("category_performance", [])
    cat_table = ""
    if cat_rows:
        cat_table = "| Category | L4W Revenue | vs Expected | % Gap |\n"
        cat_table += "|---|---|---|---|\n"
        for r in cat_rows:
            sign = "+" if r.get("gap", 0) >= 0 else ""
            cat_table += (
                f"| {r['category']} | ${r.get('actual', 0):,.0f} | "
                f"{sign}${r.get('gap', 0):,.0f} | {sign}{r.get('pct', 0):.1f}% |\n"
            )

    # ── Distribution gaps table ───────────────────────────────────────────────
    gap_rows = store_data.get("distribution_gaps", [])
    gap_table = ""
    if gap_rows:
        gap_table = "| Product | Stock Status | Peer Avg Units/Wk | Weekly Opp |\n"
        gap_table += "|---|---|---|---|\n"
        for g in gap_rows[:6]:
            gap_table += (
                f"| {g.get('product_name','')[:30]} | {g.get('stock_status','')} | "
                f"{g.get('peer_avg_units', 0):.1f} | ${g.get('weekly_opp', 0):,.2f} |\n"
            )

    prompt = f"""
You are a Kellanova NZ retail analytics system generating a professional executive store review.
Output ONLY valid Markdown — no preamble, no commentary outside the document.

FORMATTING RULES — follow exactly:
- Do NOT wrap numbers in backticks. Never use `code` formatting for any values.
- Write all dollar amounts and percentages inline in normal text.
- Use **bold** to emphasise key figures and action items.
- Tables are already provided below — copy them verbatim, then add your analysis beneath.

Use EXACTLY this structure:

# {store_data.get('chain')} Performance Review
## {store_data.get('store_name')} — {store_data.get('city')}

---

## THE SITUATION: Performance Overview

| Period | Sales | vs Expected | % vs Expected |
|---|---|---|---|
| Last 4 Weeks | ${l4w_actual:,.2f} | ${l4w_gap:+,.2f} | {l4w_pct:+.2f}% |
| Last 13 Weeks | ${l13w_actual:,.2f} | ${l13w_gap:+,.2f} | {l13w_pct:+.2f}% |
| Year to Date (est.) | ${ytd_actual:,.2f} | ${ytd_gap:+,.2f} | {ytd_pct:+.2f}% |

### Key Observations

Write exactly 4 bullet points. Each must start with a **bolded insight title** followed by one sentence of explanation. Use the numbers above. Reference trend direction, urgency, and business impact.

---

## ROOT CAUSE ANALYSIS: Why This Is Happening

### Category Performance

{cat_table if cat_table else "No category breakdown available."}

Write 3–4 bullet points identifying which categories/brands drive the gap. Each starts with a **bolded category or brand name**. Be specific about dollar amounts and percentages.

### Execution Issues

- Detected issues: {store_data.get('issues', 'None')}
- Root causes: {store_data.get('root_causes', 'N/A')}
- Nearby events: {store_data.get('nearby_events', 'None')}

Write 2–3 bullet points on execution gaps (OOS, compliance, display) and any event opportunity.

---

## ACTION PLAN: How To Close The Gap

Write 2–3 action bullets with expected impact amounts where possible.

### Distribution & Display Recommendations

{gap_table if gap_table else "No distribution gaps identified — store is well-ranged vs chain peers."}

End with one sentence on the priority recovery window.
"""
    return _call_llm(prompt, max_tokens=1200)


# ── Health check ──────────────────────────────────────────────────────────────
def check_ollama_available() -> dict:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"available": True, "models": models}
    except Exception:
        return {"available": False, "models": []}

