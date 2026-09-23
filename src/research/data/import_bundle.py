"""Opt-in, exclusive-output entrypoint for the repaired importers."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ..contracts import ContractError
from .brat import read_brat
from .coverage import map_annotations
from .tei import read_tei


def native_policy(format_name: str, length: int) -> dict:
    if format_name == "tei":
        software_types = {"software": {}}
        versions, relations = ["version"], ["version_of"]
        ignored = ["bibr", "publisher", "url", "language"]
    else:
        software_types = {f"{kind}_{use}": {} for kind in
                          ("Application", "PlugIn", "ProgrammingEnvironment", "OperatingSystem")
                          for use in ("Usage", "Creation", "Deposition", "Mention")}
        versions, relations = ["Version"], ["Version_of"]
        ignored = ["Developer", "URL", "License", "Citation", "Abbreviation", "AlternativeName", "Release"]
    return {"policy_version": f"{format_name}-positive-only-v2.0", "software_types": software_types,
            "version_types": versions, "version_relations": relations, "ignored_types": ignored,
            "coverage": [{"start": 0, "end": length, "fields": {}, "status": "partial",
                          "provenance": {"kind": "native_import", "coverage_audited": False}}]}


def write_import_bundle(files: list[Path], format_name: str, output: Path) -> dict:
    if not files:
        raise ContractError({}, "sources", "no source documents selected", "INPUT_FAILED")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    results, sources, ids = [], [], set()
    for path in files:
        data = path.read_bytes()
        source = {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
        if format_name == "tei":
            parsed = read_tei(data)
        elif format_name == "brat":
            ann_path = path.with_suffix(".ann")
            ann_bytes = ann_path.read_bytes()
            parsed = read_brat(data.decode("utf-8"), ann_bytes.decode("utf-8"))
            source.update(annotation_path=str(ann_path.resolve()),
                          annotation_sha256=hashlib.sha256(ann_bytes).hexdigest())
        else:
            raise ValueError(f"unknown native format: {format_name}")
        doc_id = f"{format_name}:{path.stem}"
        if doc_id in ids:
            raise ContractError({"document_id": doc_id}, "document_id", "duplicate source ID; select distinct documents")
        ids.add(doc_id)
        doc = {"document_id": doc_id, "text": parsed["text"],
               "metadata": {"source": str(path.resolve()), "extraction_method": f"{format_name}-v2"},
               "supplied_text_scope": "unverified", "native_split": None}
        try:
            results.append(map_annotations(parsed, doc, native_policy(format_name, len(parsed["text"]))))
        except ContractError as exc:
            for issue in exc.issues:
                issue["file"] = str(path)
            raise
        sources.append(source)
    output.mkdir(parents=True, exist_ok=False)
    files_out = {
        "documents.jsonl": [r["document"] for r in results],
        "occurrences.jsonl": [o for r in results for o in r["occurrences"]],
        "coverage.jsonl": [c for r in results for c in r["coverage"]],
        "native.jsonl": [{"document_id": r["document"]["document_id"], **r["native_annotations"]} for r in results],
        "normalization.jsonl": [{"document_id": r["document"]["document_id"], **r["normalization"]} for r in results],
        "issues.jsonl": [{"document_id": r["document"]["document_id"], "issues": r["issues"],
                          "exclusions": r["exclusions"]} for r in results],
    }
    hashes = {}
    for name, rows in files_out.items():
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(payload)
        hashes[name] = hashlib.sha256(payload.encode()).hexdigest()
    manifest = {"schema_version": "2.0", "importer_version": "native-v2.0", "format": format_name,
                "document_count": len(results), "occurrence_count": len(files_out["occurrences.jsonl"]),
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
                "output_sha256": hashes, "coverage_audited": False,
                "warning": "Positive-only import. No full-text negative coverage or intent/sentiment compatibility inferred."}
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest
