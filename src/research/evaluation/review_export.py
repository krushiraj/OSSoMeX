"""Export frozen comparison passages and references for local human review."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


DEFAULT_ARMS = ("full-label-006", "softcite-0.8.1")


def _span_matches(text, span, label):
    start, end = span["start"], span["end"]
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(text):
        raise ValueError(f"{label} span has invalid offsets: {start}:{end}")
    if "text" in span and text[start:end] != span["text"]:
        raise ValueError(f"{label} span text differs from full passage at {start}:{end}")


def _json_lines(path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid reference JSON at line {line_number}: {error}") from error
    return rows


def _checked_rows(report_path, references_path, arms):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reference_sha = hashlib.sha256(references_path.read_bytes()).hexdigest()
    if report.get("provenance", {}).get("reference_sha256") != reference_sha:
        raise ValueError("Frozen report reference SHA-256 does not match references file")

    references = {}
    for reference in _json_lines(references_path):
        document_id = reference["document_id"]
        if document_id in references:
            raise ValueError(f"Duplicate reference document_id: {document_id}")
        references[document_id] = reference

    rows = []
    seen = set()
    for document in report["documents"]:
        document_id = document["document_id"]
        if document_id in seen:
            raise ValueError(f"Duplicate report document_id: {document_id}")
        seen.add(document_id)
        if document_id not in references:
            raise ValueError(f"Missing reference document_id: {document_id}")
        reference = references[document_id]
        text = document["text"]
        revision = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        if document["text_revision"] != revision or reference["text_revision"] != revision:
            raise ValueError(f"text_revision mismatch for {document_id}")

        for span in reference.get("spans", []):
            _span_matches(text, span, f"Reference {document_id}")
        for occurrence in reference.get("attribute_occurrences", []):
            name_span = occurrence.get("name_span")
            if name_span is not None:
                _span_matches(text, {**name_span, "text": occurrence["name"]},
                              f"Reference occurrence {document_id}")

        by_arm = {}
        for arm in document["arms"]:
            arm_id = arm["arm_id"]
            if arm_id in by_arm:
                raise ValueError(f"Duplicate prediction arm {arm_id} for {document_id}")
            by_arm[arm_id] = arm
        predictions = []
        for arm_id in arms:
            if arm_id not in by_arm:
                raise ValueError(f"Missing prediction arm {arm_id} for {document_id}")
            arm = by_arm[arm_id]
            for span in arm.get("spans", []):
                _span_matches(text, span, f"Prediction {arm_id} {document_id}")
            predictions.append(arm)

        rows.append({
            "document_id": document_id,
            "text": text,
            "text_revision": revision,
            "source": document.get("source"),
            "source_ids": document.get("source_ids"),
            "source_associations": document.get("source_associations"),
            "license_status": document.get("license_status"),
            "report_metadata": {key: value for key, value in document.items()
                                if key not in {"text", "arms", "training_eligible",
                                               "future_untouched_test_eligible"}} |
                               {"training_eligible": False, "future_untouched_test_eligible": False},
            "reference": reference,
            "predictions": predictions,
            "local_review_only": True,
            "human_review_pending": True,
            "training_eligible": False,
            "future_untouched_test_eligible": False,
        })
    extra = set(references) - seen
    if extra:
        raise ValueError(f"Reference has {len(extra)} document_id(s) missing from report: {sorted(extra)[0]}")
    return rows


def _fence(text):
    marker = "```"
    while marker in text:
        marker += "`"
    return marker


def _format_span(span):
    return f"{span.get('text', '')!r} [{span['start']}:{span['end']}] ({span.get('label', 'SOFTWARE')})"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _covers_full_text(regions, text_length, label="SOFTWARE"):
    intervals = sorted((region["start"], region["end"]) for region in regions
                       if region.get("label", label) == label and region.get("complete", True))
    end = 0
    for start, stop in intervals:
        if start > end:
            return False
        end = max(end, stop)
    return end >= text_length


def _known_value(value, known):
    if not known:
        return "unknown"
    if value is None or value == []:
        return "none (reviewed)"
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


def _expected_alias(reference, occurrence):
    mention_id = occurrence.get("mention_id")
    for pair in reference.get("alias_pairs", []):
        if mention_id in pair.get("member_mention_ids", []):
            members = pair.get("members", [])
            names = " ↔ ".join(f"{member['name']} [{member['span']['start']}:{member['span']['end']}]"
                                for member in members)
            return f"{pair.get('decision', 'pair')} ({'known' if pair.get('known') else 'unknown'}): {names}"
    return "unknown (no reviewed pair)"


def _expected_versions(occurrence):
    if not (occurrence.get("known") or {}).get("versions"):
        return "unknown"
    links = occurrence.get("version_links") or []
    if not links:
        return "none (reviewed)"
    return ", ".join(f"{link['text']} [{link['span']['start']}:{link['span']['end']}]"
                     for link in links)


def _edge_text(edge):
    software, version = edge["software"], edge["version"]
    return (f"{software['text']} [{software['start']}:{software['end']}] → "
            f"{version['text']} [{version['start']}:{version['end']}]")


def _predicted_versions(arm, span):
    edges = [edge for edge in arm.get("version_links", [])
             if edge["software"]["start"] == span["start"]
             and edge["software"]["end"] == span["end"]]
    if not edges:
        return "none predicted"
    return ", ".join(f"{edge['version']['text']} "
                     f"[{edge['version']['start']}:{edge['version']['end']}]" for edge in edges)


def _prediction_value(field, key):
    if field is None:
        return "not provided"
    value = field.get(key)
    if not isinstance(value, dict) or value.get("status") != "success":
        return "unknown"
    if value.get("value") in (None, []):
        return "none predicted"
    return _known_value(value["value"], True)


def _markdown(rows, report_path, references_path):
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    reference_sha = hashlib.sha256(references_path.read_bytes()).hexdigest()
    lines = ["# Local passage review", "",
             "Agent provisional references. Human review pending. Local review only; do not train on or redistribute these passages.", "",
             f"Report: `{report_path.name}` (SHA-256 `{report_sha}`)",
             f"References: `{references_path.name}` (SHA-256 `{reference_sha}`)", "",
             "Empty names under complete software coverage mean reviewed absence. Empty names without coverage mean unknown.", ""]
    for number, row in enumerate(rows, 1):
        reference = row["reference"]
        text = row["text"]
        source_ids = row.get("source_ids") or {}
        lines.extend([f"## {number}. {row['document_id']}", "",
                      f"Source IDs: `{json.dumps(source_ids, ensure_ascii=False)}`. ",
                      f"Text revision: `{row['text_revision']}`. Length: {len(text)} codepoints. "
                      f"License: {row['license_status']}. Human review: pending.", "",
                      "### Full input passage", "", _fence(text), text, _fence(text), "",
                      "### Provisional reference", ""])
        spans = reference.get("spans", [])
        if not spans:
            if _covers_full_text(reference.get("coverage", []), len(text)):
                lines.append("Reviewed absence of names across the full passage.")
            elif _covers_full_text(reference.get("ignored_software", []), len(text)):
                lines.append("Fully masked: name and version status unknown across the full passage.")
            else:
                lines.append("No accepted names in covered regions; uncovered text remains unknown.")
        lines.extend(["| Name | Offsets | Versions | Intent | Sentiment | Alias |",
                      "| --- | --- | --- | --- | --- | --- |"])
        for occurrence in reference.get("attribute_occurrences", []):
            known = occurrence.get("known") or {}
            name_span = occurrence["name_span"]
            offsets = f"{name_span['start']}:{name_span['end']}"
            intent_known = all(known.get(intent) for intent in ("created", "used", "shared"))
            values = (occurrence["name"], offsets, _expected_versions(occurrence),
                      _known_value(occurrence.get("intents"), intent_known),
                      _known_value(occurrence.get("sentiment"), known.get("sentiment")),
                      _expected_alias(reference, occurrence))
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
        if not reference.get("attribute_occurrences"):
            lines.append("| — | — | — | — | — | — |")
        lines.append("")
        lines.append("Coverage: " + ("; ".join(f"{region['label']} {region['start']}:{region['end']} "
                                              f"({'complete' if region.get('complete') else 'partial'})"
                                              for region in reference.get("coverage", [])) or "none (unknown)"))
        lines.append(f"Ignored: {len(reference.get('ignored_software', []))} software region(s), "
                     f"{len(reference.get('ignored_versions', []))} version region(s). "
                     f"Version links: {len(reference.get('version_links', []))}; "
                     f"reviewed alias pairs: {sum(bool(pair.get('known')) for pair in reference.get('alias_pairs', []))}.")
        for pair in reference.get("alias_pairs", []):
            if pair.get("known"):
                members = " ↔ ".join(f"{member['name']} [{member['span']['start']}:{member['span']['end']}]"
                                    for member in pair.get("members", []))
                lines.append(f"Reviewed alias pair ({pair.get('decision', 'unknown')}): {members}")
        lines.extend(["", "<details><summary>Show saved predictions</summary>", ""])
        for arm in row["predictions"]:
            lines.extend([f"**{arm['arm_id']}** ({arm.get('status', 'unknown')})", "",
                          "| Name | Offsets | Versions | Intent | Sentiment | Alias |",
                          "| --- | --- | --- | --- | --- | --- |"])
            fields = {(field["name_span"]["start"], field["name_span"]["end"]): field
                      for field in arm.get("fields", []) if field.get("name_span")}
            for span in arm.get("spans", []):
                if span.get("label") != "SOFTWARE":
                    continue
                field = fields.get((span["start"], span["end"]))
                values = (span.get("text", ""), f"{span['start']}:{span['end']}",
                          _predicted_versions(arm, span), _prediction_value(field, "intents"),
                          _prediction_value(field, "sentiment"),
                          "see groups" if arm.get("alias_groups") else
                          ("unsupported" if arm["arm_id"].startswith("softcite") else "none predicted"))
                lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
            if not any(span.get("label") == "SOFTWARE" for span in arm.get("spans", [])):
                lines.append("| none predicted | — | — | — | — | — |")
            version_candidates = [span for span in arm.get("spans", []) if span.get("label") == "VERSION"]
            lines.append("Version candidates: " +
                         ("; ".join(f"{span['text']} [{span['start']}:{span['end']}]"
                                    for span in version_candidates) if version_candidates else "none"))
            lines.append("Linked owners: " +
                         ("; ".join(_edge_text(edge) for edge in arm.get("version_links", []))
                          if arm.get("version_links") else "none"))
            lines.extend(["", f"Version links: {len(arm.get('version_links', []))}; "
                          f"alias groups: {len(arm.get('alias_groups', []))}.", ""])
        lines.extend(["</details>", "", "Human review: pending", ""])
    return "\n".join(lines)


def export(report_path, references_path, output_path, arms=DEFAULT_ARMS):
    report_path = Path(report_path)
    references_path = Path(references_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError(f"Output already exists: {output_path}")
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("Arms must be a nonempty list without duplicates")
    rows = _checked_rows(report_path, references_path, arms)
    markdown = _markdown(rows, report_path, references_path)
    jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    output_path.mkdir(parents=True, exist_ok=False)
    (output_path / "review.md").write_bytes(markdown.encode("utf-8"))
    (output_path / "review.jsonl").write_bytes(jsonl.encode("utf-8"))
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    args = parser.parse_args(argv)
    try:
        count = export(args.report, args.references, args.output, args.arms)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"Review export failed: {error}", file=sys.stderr)
        return 1
    print(f"Exported {count} local review passages to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
