"""Streamlit decision-support dashboard for LABrador.

Run with ``streamlit run app.py`` from the repository root.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from labrador_roi.cli import (
    DEMO_COMPARABLES_JSON,
    DEMO_PROGRAM,
    build_interpretability_manifest,
    comparable_summary,
    reality_anchor_surface,
    run_analysis,
    validate_comparables_payload,
    validate_program_payload,
)

APP_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="LABrador · Therapeutic Program Economics",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink: #172a3a; --blue: #1e5f74; --mint: #dff3ec; --sand: #fff8e9; }
      .block-container { padding-top: 1.6rem; padding-bottom: 3rem; }
      [data-testid="stSidebar"] { background: #f5f8f7; color: #172a3a; }
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #172a3a;
      }
      [data-testid="stSidebar"] button p { color: white; }
      .hero {
        padding: 1.35rem 1.55rem; border-radius: 18px;
        background: linear-gradient(120deg, #173f5f 0%, #206a78 62%, #4b9b87 100%);
        color: white; box-shadow: 0 12px 30px rgba(23, 63, 95, .15); margin-bottom: 1rem;
      }
      .hero h1 { margin: 0; font-size: 2.2rem; letter-spacing: -.03em; }
      .hero p { margin: .35rem 0 0; opacity: .9; max-width: 850px; }
      .status-warning {
        padding: .8rem 1rem; border-left: 5px solid #e39b26; border-radius: 8px;
        background: var(--sand); color: #5a3b08; margin: .65rem 0 1rem;
      }
      .status-ok {
        padding: .8rem 1rem; border-left: 5px solid #35866f; border-radius: 8px;
        background: var(--mint); color: #163f34; margin: .65rem 0 1rem;
      }
      .eyebrow { color: #4b6876; font-size: .78rem; letter-spacing: .08em;
        text-transform: uppercase; font-weight: 700; }
      div[data-testid="stMetric"] {
        border: 1px solid #dfe8e5; padding: .8rem 1rem; border-radius: 12px;
        background: rgba(255,255,255,.7);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_data(item) for item in value]
    return value


def _decode_json(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8-sig"))


def _decode_comparables(raw: bytes, name: str) -> Any:
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw)).where(pd.notna, None).to_dict("records")
    return _decode_json(raw)


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _lookup(value: Any, names: tuple[str, ...]) -> Any:
    wanted = {name.lower() for name in names}
    for key, child in _walk(value):
        if key.lower() in wanted and child not in (None, ""):
            return child
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_money(value: Any, currency: str = "USD") -> str:
    amount = _as_float(value)
    if amount is None:
        return "—"
    symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency.upper(), f"{currency} ")
    absolute = abs(amount)
    if absolute >= 1_000_000_000:
        return f"{symbol}{amount / 1_000_000_000:,.2f}B"
    if absolute >= 1_000_000:
        return f"{symbol}{amount / 1_000_000:,.2f}M"
    if absolute >= 1_000:
        return f"{symbol}{amount / 1_000:,.1f}K"
    return f"{symbol}{amount:,.0f}"


def _format_number(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "—"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:,.2f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:,.1f}K"
    return f"{number:,.1f}"


def _records(value: Any) -> list[dict[str, Any]]:
    value = _data(value)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("comparables", "drugs", "items"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _flatten_comparables(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        item = {key: value for key, value in row.items() if key not in {"price", "evidence"}}
        price = row.get("price") if isinstance(row.get("price"), dict) else {}
        item.update(
            {
                "price_amount": price.get("amount"),
                "price_currency": price.get("currency"),
                "price_basis": price.get("basis"),
                "price_period": price.get("period"),
                "price_year": price.get("price_year"),
                "price_evidence_grade": (
                    price.get("evidence", {}).get("grade")
                    if isinstance(price.get("evidence"), dict)
                    else None
                ),
            }
        )
        flattened.append(item)
    return flattened


def _first_table(value: Any, required_tokens: tuple[str, ...]) -> list[dict[str, Any]]:
    """Find a result table whose columns resemble the requested concepts."""

    if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
        columns = {str(key).lower() for row in value for key in row}
        if all(any(token in column for column in columns) for token in required_tokens):
            return value
    if isinstance(value, dict):
        for child in value.values():
            found = _first_table(child, required_tokens)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _first_table(child, required_tokens)
            if found:
                return found
    return []


def _column(frame: pd.DataFrame, tokens: tuple[str, ...]) -> str | None:
    for token in tokens:
        for column in frame.columns:
            if token in str(column).lower():
                return str(column)
    return None


def _contains_synthetic(value: Any) -> bool:
    for key, child in _walk(value):
        if key.lower() in {"synthetic", "is_synthetic"} and (
            child is True or str(child).lower() == "true"
        ):
            return True
        if key.lower() in {"decision_grade", "evidence_grade", "source_type"} and (
            "synthetic" in str(child).lower() or "not_decision_grade" in str(child).lower()
        ):
            return True
    return False


def _warnings(result: dict[str, Any]) -> list[str]:
    value = result.get("warnings")
    if isinstance(value, list):
        rendered = []
        for item in value:
            if isinstance(item, dict):
                code = item.get("code", "WARNING")
                message = item.get("message", item)
                rendered.append(f"{code}: {message}")
            else:
                rendered.append(str(item))
        return rendered
    if value:
        return [str(value)]
    return []


@st.cache_data(show_spinner=False)
def _bundled_payloads() -> tuple[Any, Any]:
    return (
        json.loads(DEMO_PROGRAM.read_text(encoding="utf-8")),
        json.loads(DEMO_COMPARABLES_JSON.read_text(encoding="utf-8")),
    )


with st.sidebar:
    st.markdown("### Analysis workspace")
    source = st.radio(
        "Input source",
        ("Bundled synthetic demo", "Upload my inputs"),
        help="The bundled demo is intentionally synthetic and never decision-grade.",
    )
    program_payload: Any | None = None
    comparable_payload: Any | None = None
    input_error: Exception | None = None
    try:
        if source == "Bundled synthetic demo":
            program_payload, comparable_payload = _bundled_payloads()
        else:
            program_file = st.file_uploader("Program JSON", type=("json",))
            comparable_file = st.file_uploader("Comparables CSV or JSON", type=("csv", "json"))
            if program_file and comparable_file:
                program_payload = _decode_json(program_file.getvalue())
                comparable_payload = _decode_comparables(
                    comparable_file.getvalue(), comparable_file.name
                )
    except Exception as exc:
        input_error = exc

    st.divider()
    simulations = st.slider("Simulation draws", 100, 5_000, 1_000, 100)
    seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42)
    run_reality_checks = st.checkbox(
        "Run cited RA/I&I plausibility checks",
        value=False,
        help=(
            "Runs a separate reality-anchor harness. A result inside a cited band is a range "
            "check, not validation of this program or model."
        ),
    )
    analyze = st.button(
        "Run analysis",
        type="primary",
        width="stretch",
        disabled=program_payload is None or comparable_payload is None,
    )
    st.caption(
        "Same seed + same input snapshot → same modeled outputs. Replay checks the saved artifact; "
        "timestamps still differ."
    )

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow" style="color:#cbe9df">LABrador · Launch, access &amp; benefit</div>
      <h1>Therapeutic Program Economics, with the assumptions showing</h1>
      <p>Connect comparable evidence to value, access, affordability and protected cash flow.
      Model outputs, cited anchors, configuration checks and falsification controls stay
      distinct.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if input_error is not None:
    st.error(f"Could not read the uploaded input: {input_error}")

if source == "Bundled synthetic demo":
    st.markdown(
        """
        <div class="status-warning"><strong>SYNTHETIC · NOT_DECISION_GRADE</strong><br>
        Every bundled program, comparable, price and outcome is fictitious. The demo proves
        workflow behavior only.</div>
        """,
        unsafe_allow_html=True,
    )

if analyze:
    try:
        validated_program = validate_program_payload(program_payload)
        validated_comparables = validate_comparables_payload(comparable_payload)
        with st.spinner("Running transparent scenario analysis…"):
            analysis_result = run_analysis(
                validated_program,
                validated_comparables,
                simulations=int(simulations),
                seed=int(seed),
            )
            anchor_report = reality_anchor_surface(enabled=run_reality_checks)
        analysis_data = _data(analysis_result)
        st.session_state["analysis_result"] = analysis_data
        st.session_state["program"] = _data(validated_program)
        st.session_state["comparables"] = _data(validated_comparables)
        st.session_state["interpretability"] = build_interpretability_manifest(
            analysis_data,
            reality_report=anchor_report,
        )
    except Exception as exc:
        st.error(f"Analysis failed validation or execution: {exc}")

result = st.session_state.get("analysis_result")
program = st.session_state.get("program")
comparables = st.session_state.get("comparables")
interpretability = st.session_state.get("interpretability")

if result is None:
    st.info("Choose inputs and run the analysis. The bundled demo is ready immediately.")
    with st.expander("What the five views answer", expanded=True):
        st.markdown(
            """
            - **Executive:** What is the screening-grade recommendation and why?
            - **Price & Comparables:** Which evidence anchors the price range?
            - **Access & Affordability:** How many patients can reach treatment, and at what cost?
            - **Cash Flow:** When does protected cash flow arrive, and what assumptions drive it?
            - **Sensitivity / Audit:** Which inputs matter and can an agent reproduce the result?
            """
        )
    st.stop()

if not isinstance(interpretability, dict):
    interpretability = build_interpretability_manifest(result)

currency = str(_lookup(program, ("currency",)) or "USD")
synthetic = (
    _contains_synthetic(program) or _contains_synthetic(comparables) or _contains_synthetic(result)
)
decision_grade = str(_lookup(result, ("decision_grade", "grade")) or "SOURCE_DEPENDENT")
if synthetic:
    decision_grade = "NOT_DECISION_GRADE"

status_class = "status-warning" if decision_grade != "DECISION_GRADE" else "status-ok"
st.markdown(
    f'<div class="{status_class}"><strong>{decision_grade}</strong> · '
    "Evidence-gate status for a screening model—not independent validation or medical, "
    "reimbursement, legal, investment, or patent advice.</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(
    (
        "Executive",
        "Price & Comparables",
        "Access & Affordability",
        "Cash Flow",
        "Sensitivity / Audit",
    )
)

with tabs[0]:
    st.markdown("#### Decision snapshot")
    st.caption(
        "MODEL OUTPUT · Calculated from the submitted snapshot and assumptions. These values are "
        "not observed outcomes or independently validated forecasts."
    )
    metric_columns = st.columns(4)
    summary_data = result.get("summary", {})
    pricing_data = result.get("pricing", [])
    first_corridor = pricing_data[0].get("annual_net_price_corridor") if pricing_data else None
    price = first_corridor.get("selected_annual_net_price") if first_corridor else None
    cash_rows_exact = result.get("cash_flow", {}).get("annual_cash_flows", [])
    patients = max(
        (
            float(row.get("initial_active_patients", 0))
            + float(row.get("expansion_active_patients", 0))
            for row in cash_rows_exact
        ),
        default=0,
    )
    npv = summary_data.get("p50_rnpv")
    probability = summary_data.get("probability_positive_rnpv")
    price_context = interpretability.get("price_context", [])
    selected_price_context = price_context[0] if price_context else {}
    price_basis = str(selected_price_context.get("price_basis") or "UNSPECIFIED")
    valuation_year = selected_price_context.get("valuation_year")
    metric_columns[0].metric(
        f"Modeled annual {price_basis}",
        _format_money(price, currency),
        help=(
            f"{currency}; valuation year {valuation_year or '—'}. See Price & Comparables for "
            "anchor years and bases."
        ),
    )
    metric_columns[1].metric("Peak active patients", _format_number(patients))
    metric_columns[2].metric("P50 risk-adjusted NPV", _format_money(npv, currency))
    metric_columns[3].metric(
        "Positive-value probability",
        f"{100 * float(probability):.1f}%" if _as_float(probability) is not None else "—",
    )

    warning_items = _warnings(result)
    if warning_items:
        st.markdown("#### Material warnings")
        st.caption(f"{len(warning_items)} engine warning(s); the first 10 are shown below.")
        for warning in warning_items[:10]:
            st.warning(warning)
        if len(warning_items) > 10:
            with st.expander("Show every warning"):
                for warning in warning_items:
                    st.write(f"- {warning}")
    else:
        st.caption("No engine warnings were returned. Source-policy caveats still apply.")

    decision_status = interpretability.get("decision_status", {})
    critical_gaps = decision_status.get("critical_evidence_gaps", [])
    if critical_gaps:
        with st.expander(f"Critical evidence gaps ({len(critical_gaps)})"):
            st.caption(
                "These are implemented evidence-gate failures. Clearing them does not by itself "
                "validate the economic model."
            )
            st.code("\n".join(str(item) for item in critical_gaps), language=None)

    summary = _lookup(result, ("executive_summary", "summary", "recommendation"))
    if isinstance(summary, str):
        st.markdown("#### Interpretation")
        st.write(summary)
    with st.expander("Exact input snapshot used for this run"):
        st.caption(f"Digest: {interpretability.get('input_record', {}).get('input_digest') or '—'}")
        st.json(result.get("input_snapshot", {"program": program, "comparables": comparables}))

with tabs[1]:
    st.markdown("#### Comparable evidence — price types stay separate")
    st.caption(
        "Every amount below must be read with its currency, price basis, price year and valuation "
        "year. An ESTIMATED_NET output is a modeled scenario, not an observed contract price."
    )
    if price_context:
        st.dataframe(pd.DataFrame(price_context), width="stretch", hide_index=True)
    corridor_rows = []
    for snapshot in result.get("pricing", []):
        corridor = snapshot.get("annual_net_price_corridor")
        if corridor:
            corridor_rows.append(
                {
                    "indication_id": snapshot.get("indication_id"),
                    "commercial_floor": corridor.get("commercial_floor"),
                    "selected_net_price": corridor.get("selected_annual_net_price"),
                    "value_ceiling": corridor.get("value_based_ceiling"),
                    "payer_affordability_ceiling": corridor.get("payer_affordability_ceiling"),
                    "price_basis": corridor.get("basis"),
                }
            )
    if corridor_rows:
        corridor_frame = pd.DataFrame(corridor_rows)
        corridor_tidy = corridor_frame.melt(
            id_vars=("indication_id", "price_basis"),
            value_vars=("commercial_floor", "selected_net_price", "value_ceiling"),
            var_name="Price layer",
            value_name=f"Annual amount ({currency})",
        )
        corridor_figure = px.bar(
            corridor_tidy,
            x="indication_id",
            y=f"Annual amount ({currency})",
            color="Price layer",
            barmode="group",
            template="plotly_white",
        )
        st.plotly_chart(corridor_figure, width="stretch")
        st.dataframe(corridor_frame, width="stretch", hide_index=True)

    comp_rows = _records(comparables)
    if comp_rows:
        comp_frame = pd.DataFrame(_flatten_comparables(comp_rows))
        name_col = _column(comp_frame, ("name", "product", "drug"))
        price_col = _column(comp_frame, ("annual_price", "price", "amount"))
        type_col = _column(comp_frame, ("price_type", "price_basis"))
        currency_col = _column(comp_frame, ("price_currency", "currency"))
        if name_col and price_col:
            comp_frame[price_col] = pd.to_numeric(comp_frame[price_col], errors="coerce")
            grouping_columns = [column for column in (currency_col, type_col) if column is not None]
            grouped = (
                comp_frame.groupby(grouping_columns, dropna=False, sort=True)
                if grouping_columns
                else [((), comp_frame)]
            )
            for group_key, group in grouped:
                key_values = group_key if isinstance(group_key, tuple) else (group_key,)
                labels_by_column = dict(zip(grouping_columns, key_values, strict=False))
                group_currency = str(labels_by_column.get(currency_col, "unspecified"))
                group_basis = str(labels_by_column.get(type_col, "unspecified"))
                st.markdown(f"**{group_currency} · {group_basis}**")
                fig = px.bar(
                    group.dropna(subset=[price_col]),
                    x=name_col,
                    y=price_col,
                    labels={
                        price_col: f"Observed input ({group_currency})",
                        name_col: "Comparable",
                    },
                    template="plotly_white",
                )
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("Validated comparables do not expose a chartable name/price pair.")
        st.dataframe(comp_frame, width="stretch", hide_index=True)
        st.json(comparable_summary(comparables), expanded=False)
    st.warning(
        "Public reimbursement, acquisition-cost, list, and synthetic values are not proof of "
        "an actual confidential manufacturer net price. A comparable range check is not price "
        "validation."
    )

with tabs[2]:
    st.markdown("#### Access funnel and payer affordability")
    access_rows = _first_table(result, ("year", "patient"))
    if access_rows:
        access_frame = pd.DataFrame(access_rows)
        year_col = _column(access_frame, ("year",))
        patient_cols = [
            column
            for column in access_frame.columns
            if any(token in str(column).lower() for token in ("eligible", "treated", "patient"))
            and column != year_col
        ]
        if year_col and patient_cols:
            tidy = access_frame.melt(
                id_vars=year_col,
                value_vars=patient_cols,
                var_name="Population stage",
                value_name="Patients",
            )
            fig = px.line(
                tidy,
                x=year_col,
                y="Patients",
                color="Population stage",
                markers=True,
                template="plotly_white",
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(access_frame, width="stretch", hide_index=True)
    else:
        st.info("No annual patient table was returned; inspect the structured audit output.")

    selected_access: dict[str, Any] = {}
    if pricing_data and first_corridor:
        selected_price = float(first_corridor["selected_annual_net_price"])
        estimates = pricing_data[0].get("access_estimates", [])
        if estimates:
            selected_access = min(
                estimates,
                key=lambda item: abs(float(item["annual_net_price"]) - selected_price),
            )
    accessible = selected_access.get("accessible_patients")
    payer_paid = selected_access.get("payer_paid_per_patient")
    payer_exposure = (
        float(accessible) * float(payer_paid)
        if accessible is not None and payer_paid is not None
        else None
    )
    covered_lives = _lookup(program, ("covered_lives",))
    pmpm = (
        payer_exposure / (float(covered_lives) * 12)
        if payer_exposure is not None and _as_float(covered_lives)
        else None
    )
    affordability_columns = st.columns(4)
    affordability_columns[0].metric("Accessible patients", _format_number(accessible))
    affordability_columns[1].metric(
        "Expected patient OOP", _format_money(selected_access.get("expected_patient_oop"), currency)
    )
    affordability_columns[2].metric(
        "Simple payer exposure", _format_money(payer_exposure, currency)
    )
    affordability_columns[3].metric("Exposure PMPM", _format_money(pmpm, currency))
    oop_basis = str(selected_access.get("patient_oop_basis") or "UNKNOWN")
    if oop_basis != "EXPLICIT_ANNUAL_OOP":
        st.error(
            f"Patient OOP basis is {oop_basis}, not an explicit annual patient-liability input. "
            "Affordability outputs remain screening-only and cannot support a decision-grade "
            "patient-access claim."
        )
    st.caption(
        f"Patient OOP basis: **{oop_basis}**. Patient income is an access and out-of-pocket "
        "constraint. It does not mechanically "
        "reduce clinical value or a QALY gain. Simple exposure is not a complete budget-impact "
        "analysis and excludes treatment-mix and timing effects."
    )

with tabs[3]:
    st.markdown("#### Protected cash flow")
    patent_clock = interpretability.get("patent_clock", {})
    patent_columns = st.columns(4)
    patent_columns[0].metric("Filing year", patent_clock.get("filing_year") or "—")
    patent_columns[1].metric("Asset patent expiry", patent_clock.get("patent_expiry_year") or "—")
    patent_columns[2].metric("Initial launch", patent_clock.get("initial_launch_year") or "—")
    expansion_years = patent_clock.get("expansion_launch_years", [])
    patent_columns[3].metric(
        "Expansion launch(es)",
        ", ".join(str(year) for year in expansion_years) if expansion_years else "None",
    )
    st.info(
        "SHARED ASSET CLOCK · Filing, launch and expansion years are calendar dates. Every label "
        "shares the same asset-level expiry; expansion does not receive a new 20-year term."
    )
    cash_rows = _first_table(result, ("year", "revenue")) or _first_table(result, ("year", "cash"))
    if cash_rows:
        cash_frame = pd.DataFrame(cash_rows)
        year_col = _column(cash_frame, ("year",))
        value_columns = [
            column
            for column in cash_frame.columns
            if any(token in str(column).lower() for token in ("revenue", "cost", "cash", "profit"))
            and column != year_col
        ]
        if year_col and value_columns:
            fig = go.Figure()
            for column in value_columns:
                fig.add_trace(
                    go.Scatter(
                        x=cash_frame[year_col],
                        y=pd.to_numeric(cash_frame[column], errors="coerce"),
                        name=str(column).replace("_", " ").title(),
                        mode="lines+markers",
                    )
                )
            fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#64748b")
            fig.update_layout(
                template="plotly_white",
                yaxis_title=f"Cash flow ({currency})",
                xaxis_title="Year",
                hovermode="x unified",
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(cash_frame, width="stretch", hide_index=True)
    else:
        st.info("No annual cash-flow table was returned; inspect the structured audit output.")
    st.caption(
        "The patent term starts at filing. A label expansion does not reset the 20-year clock."
    )

with tabs[4]:
    st.markdown("#### Sensitivity and reproducibility")
    simulation_design = interpretability.get("simulation_design", {})
    simulation_columns = st.columns(3)
    simulation_columns[0].metric(
        "Seed",
        simulation_design.get("seed") if simulation_design.get("seed") is not None else "—",
    )
    simulation_columns[1].metric(
        "Draws",
        simulation_design.get("draws") if simulation_design.get("draws") is not None else "—",
    )
    simulation_columns[2].metric(
        "Sampled drivers", len(simulation_design.get("sampled_drivers", []))
    )
    st.caption(str(simulation_design.get("interpretation", "")))
    with st.expander("Exact sampled-driver and RNG contract"):
        st.markdown("**Sampled drivers and ranges**")
        st.json(simulation_design.get("assumption_ranges", {}), expanded=False)
        st.markdown("**RNG contract and correlation**")
        st.json(simulation_design.get("rng_contract", {}), expanded=False)

    sensitivity_rows = _first_table(result, ("parameter", "value")) or _first_table(
        result, ("variable", "impact")
    )
    if sensitivity_rows:
        sensitivity_frame = pd.DataFrame(sensitivity_rows)
        parameter_col = _column(sensitivity_frame, ("parameter", "variable", "input"))
        impact_col = _column(sensitivity_frame, ("impact", "delta", "value"))
        if parameter_col and impact_col:
            sensitivity_frame[impact_col] = pd.to_numeric(
                sensitivity_frame[impact_col], errors="coerce"
            )
            sensitivity_frame = sensitivity_frame.sort_values(impact_col)
            fig = px.bar(
                sensitivity_frame,
                x=impact_col,
                y=parameter_col,
                orientation="h",
                template="plotly_white",
                color=impact_col,
                color_continuous_scale="RdYlGn",
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, width="stretch")
    else:
        uncertainty = result.get("uncertainty", {})
        labels = {
            "rnpv": "Risk-adjusted NPV",
            "protected_net_revenue": "Protected net revenue",
            "post_loe_net_revenue": "Post-LOE net revenue",
            "peak_cash_at_risk": "Peak cash at risk",
        }
        interval_rows = [
            {
                "Metric": label,
                "P10": uncertainty[key]["p10"],
                "P50": uncertainty[key]["p50"],
                "P90": uncertainty[key]["p90"],
            }
            for key, label in labels.items()
            if isinstance(uncertainty.get(key), dict)
        ]
        if interval_rows:
            interval_frame = pd.DataFrame(interval_rows)
            figure = go.Figure()
            for row in interval_rows:
                figure.add_trace(
                    go.Scatter(
                        x=[row["P10"], row["P90"]],
                        y=[row["Metric"], row["Metric"]],
                        mode="lines",
                        line={"width": 9, "color": "#9bc8ba"},
                        showlegend=False,
                        hovertemplate="P10-P90: %{x:,.0f}<extra></extra>",
                    )
                )
                figure.add_trace(
                    go.Scatter(
                        x=[row["P50"]],
                        y=[row["Metric"]],
                        mode="markers",
                        marker={"size": 12, "color": "#173f5f"},
                        showlegend=False,
                        hovertemplate="P50: %{x:,.0f}<extra></extra>",
                    )
                )
            figure.update_layout(
                template="plotly_white",
                xaxis_title=f"Outcome ({currency})",
                yaxis_title=None,
            )
            st.plotly_chart(figure, width="stretch")
            st.dataframe(interval_frame, width="stretch", hide_index=True)
        else:
            st.info("No chartable sensitivity or uncertainty output was returned.")

    st.markdown("#### Internal output reconciliation")
    reconciliation = interpretability.get("output_reconciliation", {})
    reconciliation_status = str(reconciliation.get("status", "NOT_AVAILABLE"))
    if reconciliation_status == "FAIL":
        st.error(
            "Internal reconciliation failed. Do not use the headline outputs until the mismatch "
            "is investigated."
        )
    elif reconciliation_status == "PASS":
        st.success("Internal arithmetic reconciliation passed.")
    else:
        st.warning("Internal reconciliation was not available for this artifact.")
    st.caption(str(reconciliation.get("interpretation", "")))
    reconciliation_checks = reconciliation.get("checks", [])
    if isinstance(reconciliation_checks, list) and reconciliation_checks:
        st.dataframe(pd.DataFrame(reconciliation_checks), width="stretch", hide_index=True)
    elif reconciliation_checks:
        st.json(reconciliation_checks, expanded=False)

    st.markdown("#### Cited reality anchors — separate from this program")
    st.dataframe(
        pd.DataFrame(interpretability.get("status_legend", [])),
        width="stretch",
        hide_index=True,
    )
    anchor_surface = interpretability.get("reality_anchors", {})
    st.caption(str(anchor_surface.get("interpretation", "")))
    anchor_status = str(anchor_surface.get("status", "NOT_RUN"))
    anchor_report = anchor_surface.get("report")
    if anchor_status == "REPORTED" and isinstance(anchor_report, dict):
        bucket_counts = anchor_report.get("bucket_counts")
        if bucket_counts:
            st.markdown("##### Counts by honest bucket")
            st.json(bucket_counts, expanded=False)
        anchor_results = anchor_report.get("results")
        if isinstance(anchor_results, list) and anchor_results:
            st.dataframe(pd.json_normalize(anchor_results), width="stretch", hide_index=True)
        else:
            st.json(anchor_report, expanded=False)
    elif anchor_status == "NOT_RUN":
        st.info(
            "Reality checks were not run. Select the sidebar option and rerun to compare the "
            "implementation with cited RA/I&I plausibility bands."
        )
    else:
        st.warning(
            f"Reality-anchor status: {anchor_status}. "
            f"{anchor_surface.get('reason') or 'No report was returned.'}"
        )

    left, right = st.columns((1, 1))
    with left:
        st.markdown("##### Calculation record")
        st.json(result, expanded=False)
    with right:
        st.markdown("##### Reproduce or hand to an agent")
        st.code(
            "labrador analyze fixtures/demo_program.json "
            "--comparables fixtures/demo_comparables.json "
            f"--simulations {simulations} --seed {seed} --output analysis.json\n"
            "labrador replay analysis.json",
            language="bash",
        )
        download_payload = dict(result)
        download_payload["interpretability"] = interpretability
        result_json = json.dumps(download_payload, indent=2, sort_keys=True)
        st.download_button(
            "Download complete JSON",
            data=result_json,
            file_name="labrador-analysis.json",
            mime="application/json",
            width="stretch",
        )
        st.caption(
            "The JSON preserves the input snapshot, seed, draw count, warnings, provenance, "
            "assumptions and interpretation manifest. Replay verifies engine-owned fields; the "
            "presentation manifest is excluded from replay equality."
        )
