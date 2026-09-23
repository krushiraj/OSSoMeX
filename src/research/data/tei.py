"""TEI text extraction with source IDs, exact spans and explicit version links."""

import xml.etree.ElementTree as ET

from ..contracts import ContractError

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
BLOCKS = {"p", "head", "div", "ab", "item", "row", "bibl", "biblStruct", "note"}


def read_tei(xml_bytes: bytes) -> dict:
    if b"<!DOCTYPE" in xml_bytes.upper() or b"<!ENTITY" in xml_bytes.upper():
        raise ContractError({}, "xml", "DTD/entity declarations are not permitted", "UNSAFE_XML")
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ContractError({}, "xml", str(exc), "MALFORMED_XML") from exc
    local = lambda element: element.tag.rsplit("}", 1)[-1]
    text_node = next((node for node in root.iter() if local(node) == "text"), None)
    if text_node is None:
        raise ContractError({}, "xml", "missing text element", "MISSING_TEXT")
    ids = set()
    for node in root.iter():
        source_id = node.get(XML_ID) or node.get("id")
        if source_id and source_id in ids:
            raise ContractError({}, "source_id", f"duplicate ID {source_id}", "DUPLICATE_SOURCE_ID")
        if source_id:
            ids.add(source_id)
    parts, spans, relations, insertions, issues = [], [], [], [], []
    offset = 0
    last = ""

    def push(value):
        nonlocal offset, last
        if value:
            parts.append(value)
            offset += len(value)
            last = value[-1]

    def walk(node):
        kind = local(node)
        first_text = next((part for part in node.itertext() if part), "")
        if kind in BLOCKS and last and not last.isspace() and first_text and not first_text[0].isspace():
            insertions.append({"offset": offset, "text": "\n", "reason": "block_boundary"})
            push("\n")
        start = offset
        push(node.text)
        for child in node:
            walk(child)
            push(child.tail)
        if kind == "rs":
            source_id = node.get(XML_ID) or node.get("id")
            if not source_id:
                source_id = f"@rs:{start}:{offset}:{len(spans)}"
                issues.append({"code": "MISSING_SOURCE_ID", "source_id": source_id})
            spans.append({"source_id": source_id, "type": node.get("type", ""),
                          "start": start, "end": offset,
                          "segments": [{"start": start, "end": offset}],
                          "attributes": dict(node.attrib)})
            if node.get("type") == "version":
                for target in (node.get("corresp") or "").split():
                    relations.append({"kind": "version_of", "source_id": source_id,
                                      "target_id": target.lstrip("#")})

    walk(text_node)
    text = "".join(parts)
    spans.sort(key=lambda item: (item["start"], -item["end"]))
    span_ids = {s["source_id"] for s in spans}
    for span in spans:
        span["text"] = text[span["start"]:span["end"]]
    for rel in relations:
        if rel["target_id"] not in span_ids:
            issues.append({"code": "DANGLING_RELATION", **rel})
    return {"format": "tei", "text": text, "spans": spans, "relations": relations,
            "insertions": insertions, "issues": issues, "native_records": []}
