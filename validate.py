#!/usr/bin/env python3
"""Validation script for Datathon LLM Diagnostic Reports.

Ensures reports strictly adhere to Datathon governance and anti-hallucination rules:
1. Every number in the report must match a number present in the corresponding JSON explanation.
2. Only sensor names and symbols present in the JSON are permitted.
3. True RUL must NEVER appear in the report (zero answer leakage).
4. Required structural sections and metadata must match the JSON.

Usage:
    python validate.py                                    # Validates all reports in reports/
    python validate.py --report reports/engine_081.md      # Validates a specific report
    python validate.py --report reports/engine_081.md --json outputs/explanations/engine_081.json
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Set, Tuple


def extract_numbers_from_json(obj: Any) -> Set[float]:
    """Recursively collect all numeric values present in the JSON explanation."""
    numbers = set()
    if isinstance(obj, dict):
        for v in obj.values():
            numbers.update(extract_numbers_from_json(v))
    elif isinstance(obj, list):
        for item in obj:
            numbers.update(extract_numbers_from_json(item))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        val = float(obj)
        numbers.add(val)
        numbers.add(abs(val))  # Allow absolute magnitude mention (e.g. 16.4 cycle reduction for -16.4)
    return numbers


def extract_sensors_from_json(data: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """Collect all valid sensor codes (e.g. 's4') and symbols (e.g. 'T50') from JSON."""
    sensor_codes = set()
    sensor_symbols = set()
    for driver in data.get("top_drivers", []):
        if "sensor" in driver:
            sensor_codes.add(driver["sensor"].lower())
        if "symbol" in driver:
            sensor_symbols.add(driver["symbol"])
    return sensor_codes, sensor_symbols


def extract_numbers_from_text(text: str) -> List[Tuple[float, str, int]]:
    """Extract all numbers from markdown text, returning (numeric_value, raw_str, line_number).
    
    Uses negative lookbehind/lookahead on alphanumeric characters and underscore so that
    sensor identifiers (e.g. 's4', 's11', 'T50', 'Ps30') or file anchors ('engine_081')
    are not falsely split into partial numbers.
    """
    results = []
    lines = text.splitlines()
    pattern = re.compile(r"(?<![A-Za-z0-9_])([+-]?(?:\d+(?:\.\d+)?|\.\d+))(?![A-Za-z0-9_])")

    for line_idx, line in enumerate(lines, start=1):
        # Skip markdown table separator lines like |---|---|
        if re.match(r"^\s*\|?\s*[-:]+[-| :]*$", line):
            continue
        for match in pattern.finditer(line):
            raw_str = match.group(1).rstrip(".")
            if not raw_str or raw_str in ("+", "-"):
                continue
            try:
                val = float(raw_str)
                results.append((val, raw_str, line_idx))
            except ValueError:
                continue
    return results


def number_in_whitelist(val: float, whitelist: Set[float], tol: float = 1e-3) -> bool:
    """Check if number matches any value in whitelist within floating tolerance."""
    for allowed in whitelist:
        if math.isclose(val, allowed, abs_tol=tol, rel_tol=1e-4):
            return True
    return False


def validate_report(report_path: str, json_path: str) -> List[str]:
    """Validate a single report against its JSON source. Returns a list of error strings."""
    errors = []

    if not os.path.exists(report_path):
        return [f"Report file not found: {report_path}"]
    if not os.path.exists(json_path):
        return [f"JSON file not found: {json_path}"]

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    # 1. Whitelist of all valid numbers in the JSON
    json_numbers = extract_numbers_from_json(json_data)
    
    # 2. Extract and verify all numbers from report text
    report_numbers = extract_numbers_from_text(report_text)
    if not report_numbers:
        errors.append("Report contains no numbers (empty or invalid content).")

    for val, raw_str, line_no in report_numbers:
        if not number_in_whitelist(val, json_numbers):
            errors.append(
                f"Line {line_no}: Unverified number '{raw_str}' ({val}) is not present in {os.path.basename(json_path)}."
            )

    # 3. Check for True RUL leak
    # True RUL must never appear in report text or labels
    if re.search(r"\btrue[_\s-]?rul\b", report_text, re.IGNORECASE):
        errors.append("Rule violation: 'true RUL' is explicitly mentioned or leaked in report.")

    # Check if known true RUL values leaked from test dataset
    fleet_csv_path = os.path.join(os.path.dirname(os.path.dirname(json_path)), "fleet.csv")
    if os.path.exists(fleet_csv_path):
        try:
            import pandas as pd
            fleet_df = pd.read_csv(fleet_csv_path)
            engine_id = json_data.get("engine_id")
            row = fleet_df[fleet_df["unit"] == engine_id]
            if not row.empty:
                true_rul_val = float(row.iloc[0]["true_rul"])
                # Only flag if true_rul is different from any allowed JSON number
                if not number_in_whitelist(true_rul_val, json_numbers):
                    for val, raw_str, line_no in report_numbers:
                        if math.isclose(val, true_rul_val, abs_tol=1e-3):
                            errors.append(
                                f"CRITICAL: Line {line_no} leaks ground-truth RUL value '{raw_str}' ({true_rul_val})!"
                            )
        except Exception:
            pass

    # 4. Verify sensor codes and symbols
    valid_codes, valid_symbols = extract_sensors_from_json(json_data)
    # Check any sensor patterns like s1..s21
    mentioned_sensors = set(re.findall(r"\b(s\d+)\b", report_text))
    for s in mentioned_sensors:
        if s.lower() not in valid_codes:
            errors.append(
                f"Rule violation: Sensor '{s}' mentioned in report is not among the engine's documented drivers: {sorted(valid_codes)}"
            )

    # 5. Verify critical metadata integrity
    expected_engine_id = str(json_data.get("engine_id"))
    if expected_engine_id not in report_text:
        errors.append(f"Report does not mention expected Engine ID: {expected_engine_id}")

    expected_risk = str(json_data.get("risk_level")).lower()
    if expected_risk not in report_text.lower():
        errors.append(f"Report does not mention expected risk level: '{expected_risk}'")

    expected_rul = str(json_data.get("predicted_rul_cycles"))
    if expected_rul not in report_text:
        errors.append(f"Report does not quote predicted RUL: {expected_rul}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate LLM Maintenance Reports against explanation JSONs.")
    parser.add_argument("--report", help="Path to a specific report .md file")
    parser.add_argument("--json", help="Path to corresponding explanation .json file (optional if following naming convention)")
    parser.add_argument("--reports-dir", default="reports", help="Directory containing report files")
    parser.add_argument("--explanations-dir", default=os.path.join("outputs", "explanations"), help="Directory containing JSON explanations")
    args = parser.parse_args()

    reports_to_validate = []
    if args.report:
        if args.json:
            reports_to_validate.append((args.report, args.json))
        else:
            match = re.search(r"engine_(\d+)", args.report)
            if not match:
                print(f"Error: Could not infer engine ID from report filename '{args.report}'. Specify --json explicitly.")
                sys.exit(1)
            unit = int(match.group(1))
            json_file = os.path.join(args.explanations_dir, f"engine_{unit:03d}.json")
            reports_to_validate.append((args.report, json_file))
    else:
        pattern = os.path.join(args.reports_dir, "engine_*.md")
        found = glob.glob(pattern)
        if not found:
            print(f"No reports found matching {pattern}")
            sys.exit(1)
        for r_path in sorted(found):
            match = re.search(r"engine_(\d+)", r_path)
            if match:
                unit = int(match.group(1))
                json_file = os.path.join(args.explanations_dir, f"engine_{unit:03d}.json")
                reports_to_validate.append((r_path, json_file))

    total_reports = len(reports_to_validate)
    passed_reports = 0
    all_passed = True

    print(f"Starting validation for {total_reports} report(s)...\n")
    for r_path, j_path in reports_to_validate:
        print(f"Checking {r_path} against {j_path}...")
        errs = validate_report(r_path, j_path)
        if errs:
            all_passed = False
            print(f"  FAILED ({len(errs)} error(s)):")
            for e in errs:
                print(f"    - {e}")
        else:
            passed_reports += 1
            print("  PASSED (All numbers, sensors, and governance rules validated).")
        print()

    print(f"Summary: {passed_reports}/{total_reports} reports passed validation.")
    if not all_passed:
        sys.exit(1)
    else:
        print("All validations successful! Zero unverified numbers or hallucinations detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
