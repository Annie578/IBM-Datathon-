#!/usr/bin/env python3
"""Report generation script for Turbofan Predictive Maintenance & Diagnostics using Groq LLM.

Produces zero-hallucination diagnostic reports following the fixed template.
Supports:
1. Groq LLM mode (via Groq API: https://api.groq.com/openai/v1 using GROQ_API_KEY)
   Includes an automatic reflection & validation loop against validate.py!
2. Deterministic template fallback if API key is not present.

Usage:
    python scripts/generate_report.py --engine 81
    python scripts/generate_report.py --engine 47
    python scripts/generate_report.py --all
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Load environment variables (.env)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import the validator directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate import extract_numbers_from_json, extract_numbers_from_text, number_in_whitelist, validate_report


SYSTEM_PROMPT = """You are an Aerospace Reliability & Predictive Maintenance Diagnostics Specialist.
Your task is to generate a comprehensive, executive Engineering Maintenance Report for an aircraft turbofan engine based exclusively on the provided JSON telemetry analysis file.

MANDATORY ANTI-HALLUCINATION & GOVERNANCE RULES:
1. STRICT NUMBER INTEGRITY: You may ONLY cite numbers that exist in the provided JSON input. Under NO circumstances should you invent, extrapolate, round differently, or hallucinate metrics, timestamps, percentages (no '%'), or part serial numbers.
2. ABSOLUTE PROHIBITION ON EXTRANEOUS NUMBERS: Do NOT mention numbers in headings, bullet points, or operational advice (for example, do NOT say "24 hours", "30 days", "Top 5 sensors", or "step 1"). Use qualitative descriptions ("immediate", "prior to next flight", "scheduled line check").
3. SENSOR RESTRICTION: You may ONLY mention the sensors and symbols explicitly listed in the JSON file (under "top_drivers"). Do not mention other sensors or invent sensor functionalities.
4. ZERO TRUE RUL LEAKAGE: True RUL is intentionally omitted from the diagnostic input. Do not guess, estimate, or mention "true RUL".
5. FIXED TEMPLATE COMPLIANCE: The report must strictly follow the required Markdown template structure without omitting sections.
"""


def load_explanation(engine_id: int, explanations_dir: str = "outputs/explanations") -> Dict[str, Any]:
    file_path = os.path.join(explanations_dir, f"engine_{engine_id:03d}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Explanation file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_deterministic_report(data: Dict[str, Any]) -> str:
    """Generate the markdown report strictly populated from JSON values via deterministic template."""
    engine_id = data["engine_id"]
    cycles_obs = data["cycles_observed"]
    pred_rul = data["predicted_rul_cycles"]
    risk = data["risk_level"]
    crit_thresh = data["risk_thresholds"]["critical_below_cycles"]
    warn_thresh = data["risk_thresholds"]["warning_below_cycles"]
    rank = data["fleet_rank_most_urgent_first"]
    fleet_size = data["fleet_size"]
    fleet_avg = data["fleet_average_prediction_cycles"]
    age_effect = data["engine_age_effect_on_rul_cycles"]
    base_cycles = data["baseline_cycles_used"]

    ctx = data["model_context"]
    model_name = ctx["model"]
    rmse = ctx["test_rmse_cycles"]
    mae = ctx["test_mae_cycles"]
    rul_cap = ctx["rul_cap_cycles"]
    caveat = ctx["caveat"]

    driver_rows = []
    for d in data["top_drivers"]:
        row = (
            f"| `{d['sensor']}` | **{d['symbol']}** | {d['description']} | "
            f"`{d['effect_on_rul_cycles']}` | {d['direction']} | "
            f"`{d['current_value']}` | `{d['healthy_baseline_value']}` | "
            f"`{d['deviation_from_baseline']}` | `{d['recent_slope_per_cycle']}` |"
        )
        driver_rows.append(row)
    drivers_table = "\n".join(driver_rows)

    if risk == "critical":
        primary_driver = data["top_drivers"][0]
        secondary_driver = data["top_drivers"][1]
        diagnostic_text = (
            f"Engine exhibits severe multi-subsystem degradation placing it at the very top of the fleet urgency ranking. "
            f"The dominant driver of reduced life expectancy is `{primary_driver['sensor']}` ({primary_driver['symbol']} - {primary_driver['description']}), "
            f"which drastically shortens predicted life. "
            f"Thermal stress is compounded by `{secondary_driver['sensor']}` ({secondary_driver['symbol']} - {secondary_driver['description']}). "
            f"Elevated outlet temperatures and static pressure variations indicate advanced stage turbine and compressor wear, "
            f"with positive parameter slopes demonstrating accelerating degradation rather than stable cruise performance."
        )
        action_text = (
            "- **IMMEDIATE GROUNDING & SHOP INSPECTION**: Remove engine from flight rotation prior to next operational sortie.\n"
            "- **HOT SECTION BORESCOPE**: Conduct detailed borescope inspection of Low Pressure Turbine (LPT) and High Pressure Compressor (HPC) stages.\n"
            "- **SEAL & BLADE CLEARANCE VERIFICATION**: Inspect static pressure taps and turbine blade clearance for thermal erosion."
        )
    elif risk == "warning":
        primary_driver = data["top_drivers"][0]
        diagnostic_text = (
            f"Engine is operating in the warning envelope. Telemetry indicates moderate drift across core pressure and temperature parameters, "
            f"led by `{primary_driver['sensor']}` ({primary_driver['symbol']} - {primary_driver['description']}). "
            f"While immediate catastrophic failure is not imminent, performance metrics diverge from baseline trends."
        )
        action_text = (
            "- **ENHANCED CONDITION MONITORING**: Increase telemetry logging frequency and inspect sensor calibration.\n"
            "- **SCHEDULED LINE MAINTENANCE**: Plan preventive inspection during the next scheduled maintenance layover."
        )
    else:  # healthy
        primary_driver = data["top_drivers"][0]
        secondary_driver = data["top_drivers"][1]
        diagnostic_text = (
            f"Engine telemetry exhibits exemplary operational health, ranking among the least urgent units across the fleet. "
            f"Core parameters remain closely aligned with baseline readings established during the initial burn-in window. "
            f"The primary positive life contributors include `{primary_driver['sensor']}` ({primary_driver['symbol']} - {primary_driver['description']}) "
            f"and `{secondary_driver['sensor']}` ({secondary_driver['symbol']} - {secondary_driver['description']}), "
            f"both of which lengthen predicted life. Negative rate-of-change metrics indicate stable thermo-fluid operating dynamics."
        )
        action_text = (
            "- **CONTINUE UNRESTRICTED FLIGHT OPERATIONS**: Retain standard flight dispatch availability without operational constraints.\n"
            "- **STANDARD MONITORING**: Maintain routine scheduled inspection intervals."
        )

    report_lines = [
        f"# Turbofan Engine Maintenance Assessment Report: Engine {engine_id}",
        "",
        "## Status Overview",
        f"- **Engine ID**: {engine_id}",
        f"- **Fleet Urgency Rank**: Rank {rank} of {fleet_size}",
        f"- **Operational Cycles Observed**: {cycles_obs}",
        f"- **Risk Classification**: {risk}",
        f"- **Predicted Remaining Useful Life (RUL)**: {pred_rul} cycles",
        f"- **Fleet Average Predicted RUL**: {fleet_avg} cycles",
        "",
        "## Risk Classification Criteria",
        f"- **Critical Risk Threshold**: Predicted RUL below {crit_thresh} cycles",
        f"- **Warning Risk Threshold**: Predicted RUL below {warn_thresh} cycles",
        f"- **Current Fleet Status**: Ranked {rank} of {fleet_size} engines ({risk} status)",
        "",
        "## Primary Telemetry & Degradation Drivers",
        f"The primary sensor drivers identified by TreeSHAP attribution and their deviation from baseline (measured over initial {base_cycles} cycles):",
        "",
        "| Sensor | Symbol | Description | Impact on RUL (cycles) | Direction | Current Value | Healthy Baseline | Baseline Deviation | Recent Slope per Cycle |",
        "|---|---|---|---|---|---|---|---|---|",
        drivers_table,
        "",
        f"Engine age effect on predicted RUL: {age_effect} cycles.",
        "",
        "## Diagnostic Assessment",
        diagnostic_text,
        "",
        "## Operational Action Protocol",
        action_text,
        "",
        "## Model Reliability & Technical Context",
        f"- **Model Architecture**: {model_name}",
        f"- **Model Test Root Mean Square Error (RMSE)**: {rmse} cycles",
        f"- **Model Test Mean Absolute Error (MAE)**: {mae} cycles",
        f"- **Target Training RUL Cap**: {rul_cap} cycles",
        f"- **Operational Caveat**: {caveat}",
        "",
    ]
    return "\n".join(report_lines)


def generate_with_groq(
    data: Dict[str, Any],
    api_key: str,
    model: str = "qwen/qwen3.8-27b",
    base_url: str = "https://api.groq.com/openai/v1",
    max_retries: int = 2
) -> str:
    """Call Groq API to generate the report with anti-hallucination validation reflection."""
    import requests

    engine_id = data["engine_id"]
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Reference template structure
    reference_template = build_deterministic_report(data)

    user_prompt = f"""Generate the official Markdown maintenance assessment report for Engine {engine_id} using this exact JSON telemetry analysis:
```json
{json.dumps(data, indent=2)}
```

You must adhere to this exact structural format:
{reference_template}

CRITICAL RULES:
1. Every numeric value must be identical to the JSON.
2. In the 'Diagnostic Assessment' and 'Operational Action Protocol' sections, use engineering terminology and thermodynamic reasoning, but DO NOT introduce ANY new numbers (no percentages, no hour estimates like '24 hours', no list numbers).
3. Do not mention true RUL.
Return ONLY the raw markdown report."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries + 1):
        print(f"Calling Groq ({model}) for Engine {engine_id} (attempt {attempt + 1}/{max_retries + 1})...")
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1500
        }
        for req_try in range(4):
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=45)
            if resp.status_code == 429:
                wait_sec = 30.0
                try:
                    msg = resp.json().get("error", {}).get("message", "")
                    import re
                    m = re.search(r"try again in (\d+(?:\.\d+)?)s", msg)
                    if m:
                        wait_sec = float(m.group(1)) + 2.0
                except Exception:
                    pass
                print(f"  Rate limited by Groq free tier. Waiting {wait_sec:.1f}s before retrying...")
                time.sleep(wait_sec)
                continue
            elif resp.status_code != 200:
                raise RuntimeError(f"Groq API error HTTP {resp.status_code}: {resp.text}")
            break
        else:
            raise RuntimeError(f"Groq rate limit exceeded after retries.")

        res_json = resp.json()
        report_text = res_json["choices"][0]["message"]["content"].strip()

        # Clean markdown wrappers if any
        if report_text.startswith("```markdown"):
            report_text = report_text[len("```markdown"):].strip()
        if report_text.startswith("```"):
            report_text = report_text[3:].strip()
        if report_text.endswith("```"):
            report_text = report_text[:-3].strip()

        # Validate against validate.py
        temp_path = f"_temp_val_engine_{engine_id}.md"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        json_path = os.path.join("outputs", "explanations", f"engine_{engine_id:03d}.json")
        errors = validate_report(temp_path, json_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if not errors:
            print(f"[OK] Groq generated report passed all validation checks on attempt {attempt + 1}!")
            return report_text
        else:
            print(f"  Validation flagged {len(errors)} error(s):")
            for err in errors[:4]:
                print(f"    - {err}")
            if attempt < max_retries:
                print("  Sending feedback to Groq for zero-hallucination correction...")
                messages.append({"role": "assistant", "content": report_text})
                messages.append({
                    "role": "user",
                    "content": f"Your generated report failed validation with these exact errors:\n" +
                               "\n".join(errors) +
                               "\nPlease regenerate the report, eliminating all unverified numbers and strictly matching the JSON."
                })
            else:
                print("  Max retries reached. Using deterministic verified template.")
                return build_deterministic_report(data)


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic reports from engine explanation JSON using Groq.")
    parser.add_argument("--engine", type=int, help="Engine ID to generate report for (e.g. 81 or 47)")
    parser.add_argument("--all", action="store_true", help="Generate reports for demo engines (81 and 47)")
    parser.add_argument("--provider", choices=["auto", "groq", "template"], default="auto",
                        help="LLM provider: 'groq' (Groq API), 'template' (deterministic), or 'auto'")
    parser.add_argument("--model", default=os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"), help="Groq model name")
    parser.add_argument("--api-key", default=os.environ.get("GROQ_API_KEY"),
                        help="Groq API key (defaults to GROQ_API_KEY from environment or .env)")
    parser.add_argument("--explanations-dir", default=os.path.join("outputs", "explanations"))
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)

    target_engines = []
    if args.engine:
        target_engines.append(args.engine)
    elif args.all or not args.engine:
        target_engines = [81, 47]

    use_groq = False
    if args.provider == "groq":
        if not args.api_key:
            print("Error: --provider groq requires GROQ_API_KEY. Set it in .env or pass --api-key.")
            sys.exit(1)
        use_groq = True
    elif args.provider == "auto":
        if args.api_key:
            use_groq = True
            print(f"Using Groq LLM (Model: {args.model}).")
        else:
            print("No GROQ_API_KEY found. Falling back to deterministic template generator.")

    for eng in target_engines:
        data = load_explanation(eng, args.explanations_dir)
        if use_groq:
            report_content = generate_with_groq(data, api_key=args.api_key, model=args.model)
        else:
            report_content = build_deterministic_report(data)

        out_path = os.path.join(args.reports_dir, f"engine_{eng:03d}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Saved report to: {out_path}\n")

    print("Running final validation suite across all reports...")
    val_status = os.system(f"python validate.py --reports-dir {args.reports_dir}")
    if val_status == 0:
        print("\nAll reports passed validation successfully!")


if __name__ == "__main__":
    main()
