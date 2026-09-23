"""Parse BRAT standoff records without first-substring alignment guesses."""

from ..contracts import ContractError, check_span


def read_brat(text: str, annotation_text: str) -> dict:
    spans, relations, records, issues = [], [], [], []
    seen = set()
    for number, line in enumerate(annotation_text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        source_id = parts[0]
        if source_id in seen and source_id != "*":
            raise ContractError({}, f"line:{number}", f"duplicate ID {source_id}", "DUPLICATE_SOURCE_ID")
        seen.add(source_id)
        records.append({"source_id": source_id, "line": number, "raw": line})
        try:
            if source_id.startswith("T"):
                type_name, boundaries = parts[1].split(" ", 1)
                segments = []
                for segment in boundaries.split(";"):
                    start, end = map(int, segment.split())
                    span = {"start": start, "end": end}
                    check_span(span, {}, f"line:{number}", len(text))
                    if segments and start < segments[-1]["end"]:
                        raise ValueError("discontinuous segments must be ordered and disjoint")
                    segments.append(span)
                value = parts[2]
                expected = " ".join(text[s["start"]:s["end"]] for s in segments)
                if expected != value:
                    issues.append({"source_id": source_id, "line": number, "code": "ALIGNMENT_UNRESOLVED"})
                spans.append({"source_id": source_id, "type": type_name, "text": value,
                              "start": segments[0]["start"], "end": segments[-1]["end"],
                              "segments": segments, "line": number})
            elif source_id.startswith("R"):
                kind, *arguments = parts[1].split()
                args = dict(a.split(":", 1) for a in arguments)
                if len(args) != 2 or len(args) != len(arguments):
                    raise ValueError("binary relation requires two unique argument roles")
                relations.append({"relation_id": source_id, "kind": kind,
                                  "source_id": args.get("Arg1"), "target_id": args.get("Arg2"),
                                  "arguments": args})
            else:
                if len(parts) < 2:
                    raise ValueError("missing TAB-separated annotation body")
                issues.append({"code": "NATIVE_RECORD_UNMAPPED", "source_id": source_id, "line": number})
        except (ValueError, IndexError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError({}, f"line:{number}", str(exc), "MALFORMED_BRAT") from exc
    span_ids = {s["source_id"] for s in spans}
    for rel in relations:
        if any(target not in span_ids for target in rel["arguments"].values()):
            issues.append({"code": "DANGLING_RELATION", **rel})
    return {"format": "brat", "text": text, "spans": spans, "relations": relations,
            "issues": issues, "native_records": records, "insertions": []}
