from __future__ import annotations

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET


LAYOUTS = ("score_only", "tab_only", "score_tab")


def _set_staff_type(staff: ET.Element, layout: str) -> None:
    staff_type = staff.find("StaffType")
    if staff_type is None:
        staff_type = ET.SubElement(staff, "StaffType")
    name = staff_type.find("name")
    if name is None:
        name = ET.SubElement(staff_type, "name")
    if layout == "score_only":
        staff_type.set("group", "pitched")
        name.text = "stdNormal"
    else:
        staff_type.set("group", "tablature")
        name.text = "tab6StrCommon"
        default_clef = staff.find("defaultClef")
        if default_clef is not None:
            staff.remove(default_clef)


def _remove_eids(element: ET.Element) -> None:
    for parent in element.iter():
        for child in list(parent):
            if child.tag == "eid":
                parent.remove(child)


def _remove_duplicate_annotations(staff: ET.Element) -> None:
    annotation_tags = {
        "Dynamic",
        "Fermata",
        "Harmony",
        "Marker",
        "RehearsalMark",
        "StaffText",
        "Tempo",
        "VBox",
    }
    for parent in staff.iter():
        for child in list(parent):
            if child.tag in annotation_tags:
                parent.remove(child)


def convert_mscx_layout(source: Path, output: Path, layout: str) -> None:
    if layout not in LAYOUTS:
        raise ValueError(f"Unsupported MuseScore layout: {layout}")
    tree = ET.parse(source)
    root = tree.getroot()
    score = root.find("Score")
    if score is None:
        raise ValueError(f"No Score element in {source}")
    parts = score.findall("Part")
    body_staves = score.findall("Staff")
    if len(parts) != 1 or len(body_staves) != 1:
        raise ValueError(
            "Automatic MuseScore layout conversion currently requires one part and one staff"
        )
    part_staff = parts[0].find("Staff")
    if part_staff is None:
        raise ValueError(f"Part has no Staff definition in {source}")

    if layout in {"score_only", "tab_only"}:
        _set_staff_type(part_staff, layout)
    else:
        tab_definition = copy.deepcopy(part_staff)
        _set_staff_type(tab_definition, "tab_only")
        _remove_eids(tab_definition)
        part_children = list(parts[0])
        insert_at = part_children.index(part_staff) + 1
        parts[0].insert(insert_at, tab_definition)

        tab_staff = copy.deepcopy(body_staves[0])
        tab_staff.set("id", "2")
        _remove_eids(tab_staff)
        _remove_duplicate_annotations(tab_staff)
        score_children = list(score)
        score.insert(score_children.index(body_staves[0]) + 1, tab_staff)

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="UTF-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a single-staff MuseScore MSCX file to score, TAB, or score+TAB."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layout", choices=LAYOUTS, required=True)
    args = parser.parse_args()
    convert_mscx_layout(args.input.resolve(), args.output.resolve(), args.layout)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
