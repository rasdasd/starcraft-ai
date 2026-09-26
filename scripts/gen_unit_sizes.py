#!/usr/bin/env python3
"""Build python/bwbot/data/unit_sizes.json from Liquipedia's "List of Unit and Building Sizes".

    python scripts/gen_unit_sizes.py            # parse the saved wikitext snapshot
    python scripts/gen_unit_sizes.py --fetch    # refresh the snapshot from Liquipedia first

Keys are BWAPI UnitType names. Each entry also carries BWAPI's own pixel extents (left, up, right,
down from the unit's centre; wsl/bwapi's UnitType.cpp when present, else the `--dims` file) and
the script checks Liquipedia against them:
    unit width  = left + right + 1,          height = up + down + 1
    gap left    = box_w / 2 - left,          gap right  = box_w / 2 - right - 1
    gap top     = box_h / 2 - up,            gap bottom = box_h / 2 - down - 1
Liquipedia content is CC BY-SA 3.0; the JSON records the source.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from bwbot import UnitType  # noqa: E402

URL = "https://liquipedia.net/starcraft/List_of_Unit_and_Building_Sizes"
DATA = ROOT / "python" / "bwbot" / "data"
WIKI = DATA / "unit_sizes.wiki"
OUT = DATA / "unit_sizes.json"
BWAPI_SRC = ROOT / "wsl" / "bwapi" / "bwapi" / "BWAPILIB" / "Source" / "UnitType.cpp"

# Liquipedia name -> BWAPI name, where the generic rules below don't find it.
ALIASES = {
    "Spider Mine": "Terran_Vulture_Spider_Mine",
    "Terran Siege Tank (Tank Mode)": "Terran_Siege_Tank_Tank_Mode",
    "Terran Siege Tank (Siege Mode)": "Terran_Siege_Tank_Siege_Mode",
    "Infested Terran": "Zerg_Infested_Terran",
    "Mutalisk Cocoon": "Zerg_Cocoon",
    "Lurker Egg": "Zerg_Lurker_Egg",
    "Data Disc": "Powerup_Data_Disk",
    "Protoss Vespene Gas Orb Type 1": "Powerup_Protoss_Gas_Orb_Type_1",
    "Protoss Vespene Gas Orb Type 2": "Powerup_Protoss_Gas_Orb_Type_2",
    "Zerg Vespene Gas Sac Type 1": "Powerup_Zerg_Gas_Sac_Type_1",
    "Zerg Vespene Gas Sac Type 2": "Powerup_Zerg_Gas_Sac_Type_2",
    "Terran Vespene Gas Tank Type 1": "Powerup_Terran_Gas_Tank_Type_1",
    "Terran Vespene Gas Tank Type 2": "Powerup_Terran_Gas_Tank_Type_2",
    "Infested Kerrigan": "Hero_Infested_Kerrigan",
    "Zerg Queen's Nest": "Zerg_Queens_Nest",
    "Mineral Field (Type 1)": "Resource_Mineral_Field",
    "Mineral Field (Type 2)": "Resource_Mineral_Field_Type_2",
    "Mineral Field (Type 3)": "Resource_Mineral_Field_Type_3",
    "Khaydarin Crystal Formation": "Special_Khaydarin_Crystal_Form",
    "Xel`Naga Temple": "Special_XelNaga_Temple",
    "Norad II (Crashed)": "Special_Crashed_Norad_II",
    "Zerg Overmind (With Shell)": "Special_Overmind_With_Shell",
    "Zerg Overmind": "Special_Overmind",
    "Zerg Cerebrate": "Special_Cerebrate",
    "Zerg Cerebrate Daggoth": "Special_Cerebrate_Daggoth",
    "Stasis Cell/Prison": "Special_Stasis_Cell_Prison",
    "Left Upper Level Door": "Special_Upper_Level_Door",
    "Left Pit Door": "Special_Pit_Door",
    "Left Wall Missile Trap": "Special_Wall_Missile_Trap",
    "Left Wall Flame Trap": "Special_Wall_Flame_Trap",
}
PREFIXES = ("", "Special_", "Powerup_", "Critter_", "Resource_", "Spell_", "Terran_", "Zerg_", "Protoss_")


def fetch() -> None:
    req = urllib.request.Request(URL + "?action=raw", headers={"User-Agent": "starcraft-ai unit sizes (research)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        WIKI.write_bytes(r.read())
    print(f"saved {WIKI}")


def bwapi_name(label: str, names: dict[str, str]) -> str | None:
    if label in ALIASES:
        return ALIASES[label]
    base = re.sub(r"\s*\(.*?\)", "", label).strip()           # "Rhynadon (Badlands Critter)" -> "Rhynadon"
    key = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower()
    for p in PREFIXES:
        hit = names.get((p + key).lower())
        if hit:
            return hit
    return None


def bwapi_dims() -> dict[int, tuple[int, int, int, int, int, int]]:
    """Unit type id -> (tile_w, tile_h, left, up, right, down) from BWAPI's source, if checked out."""
    if not BWAPI_SRC.exists():
        return {}
    text = BWAPI_SRC.read_text(encoding="utf-8", errors="ignore")
    block = text[text.index("unitDimensions[UnitTypes::Enum::MAX]"):]
    block = block[block.index("{") + 1:block.index("};")]
    rows = re.findall(r"\{(\s*\d+\s*(?:,\s*\d+\s*){5})\}", block)
    return {i: tuple(int(v) for v in r.split(",")) for i, r in enumerate(rows)}


def parse(text: str):
    """(section, label, cells) for every table row: sections are unit tables or building tables."""
    section = kind = None
    for line in text.splitlines():
        h = re.match(r"^=+\s*(.*?)\s*=+$", line)
        if h:
            section = h.group(1)
            continue
        if line.startswith("!"):
            head = line.lstrip("!").split("||")[0].strip()
            if head != "Size":                 # unit tables put "!Size" on its own header line
                kind = {"Ground Unit": "ground", "Air Unit": "air", "Unit": "unit", "Add-on": "addon",
                        "Resource": "resource"}.get(head, "building")
            continue
        if not line.startswith("|") or line.startswith(("|-", "|}", "{|")):
            continue
        cells = [c.strip() for c in line[1:].split("||")]
        yield section, kind, cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download the wikitext snapshot first")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    names = {u.name.lower(): u.name for u in UnitType}
    dims = bwapi_dims()
    units, buildings, unknown, mismatch = {}, {}, [], []
    pending = None                     # unit tables put the label and the size on separate lines
    for section, kind, cells in parse(WIKI.read_text(encoding="utf-8")):
        if kind in ("ground", "air", "unit"):
            if len(cells) == 1 and pending is None:
                pending = cells[0]
                continue
            label, size = (pending, cells[0]) if pending is not None else (cells[0], cells[1])
            pending = None
            label = label.rstrip(":").strip()
            w, h = (int(v) for v in size.lower().split("x"))
            name = bwapi_name(label, names)
            if name is None:
                unknown.append(label)
                continue
            entry = {"liquipedia": label, "section": section, "kind": kind, "width": w, "height": h}
            d = dims.get(int(UnitType[name]))
            if d:
                entry["bwapi"] = list(d[2:])
                if (d[2] + d[4] + 1, d[3] + d[5] + 1) != (w, h):
                    mismatch.append(f"{name}: liquipedia {w}x{h}, bwapi {d[2] + d[4] + 1}x{d[3] + d[5] + 1}")
            units[name] = entry
        else:
            label = cells[0].rstrip(":").strip()
            top, left, right, bottom = (int(v) for v in cells[1:5])
            bw, bh = (int(v) for v in cells[5].split("x"))
            rw, rh = (int(v) for v in cells[6].split("x"))
            name = bwapi_name(label, names)
            if name is None:
                unknown.append(label)
                continue
            entry = {"liquipedia": label, "section": section, "kind": kind, "box": [bw, bh], "real": [rw, rh],
                     "gap": {"top": top, "left": left, "right": right, "bottom": bottom}}
            d = dims.get(int(UnitType[name]))
            if d:
                entry["bwapi"] = list(d[2:])
                want = (bh // 2 - d[3], bw // 2 - d[2], bw // 2 - d[4] - 1, bh // 2 - d[5] - 1)
                if want != (top, left, right, bottom):
                    mismatch.append(f"{name}: liquipedia gaps t/l/r/b {top}/{left}/{right}/{bottom}, "
                                    f"bwapi {'/'.join(map(str, want))}")
            buildings[name] = entry
    out = {
        "source": {"url": URL, "retrieved": str(date.fromtimestamp(WIKI.stat().st_mtime)),
                   "license": "CC BY-SA 3.0 (Liquipedia)"},
        "notes": ["Pixels. units: width x height of the collision box.",
                  "buildings: box = tile footprint in pixels, real = collision box, gap = free pixels between "
                  "the collision box and the footprint edge on each side. Two buildings in adjacent tiles leave "
                  "left.gap.right + right.gap.left pixels between them; a unit passes a gap at least its width "
                  "(vertical passage) or height (horizontal passage).",
                  "Add-on gaps are relative to the add-on's own footprint; a negative left gap overlaps the "
                  "parent's gap to seal it.",
                  "bwapi: [left, up, right, down] extents from the unit's centre, as BWAPI reports them "
                  "(UnitType::dimensionLeft() ...)."],
        "units": units,
        "buildings": buildings,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(units)} units, {len(buildings)} buildings"
          + (f", checked against BWAPI ({len(mismatch)} mismatches)" if dims else " (no BWAPI source to check)"))
    for m in mismatch:
        print("  mismatch:", m)
    if unknown:
        print("  no BWAPI name for:", ", ".join(unknown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
