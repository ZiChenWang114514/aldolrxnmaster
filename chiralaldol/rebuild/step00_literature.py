"""Step 0: SA convention literature confirmation.

Confirmed finding (Evans JACS 1981;103:2127, Zimmerman-Traxler 1957):
  - Syn: OH and alpha-substituent on SAME side in zig-zag projection
  - Anti: OH and alpha-substituent on OPPOSITE sides
  - This is Masamune-Heathcock convention (relative, NOT absolute R/S)
  - Z-enolate → syn product via Zimmerman-Traxler chair TS
  - The mapping Ca_label ↔ CIP(R/S) is substrate-dependent and must be
    determined empirically from the data via majority vote.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SA_CONVENTION = {
    "syn_definition": "OH and alpha-substituent on same side in zig-zag projection",
    "anti_definition": "OH and alpha-substituent on opposite sides in zig-zag projection",
    "convention": "Masamune-Heathcock (replaces erythro/threo)",
    "key_insight": "syn/anti is relative stereochemistry, NOT equivalent to same/different CIP R/S",
    "selectivity_rule": "Z-enolate → syn product via Zimmerman-Traxler chair TS",
    "evans_specifics": "Bu2BOTf/Et3N → >98% Z-enolate → syn; oxazolidinone controls enantioface",
    "label_mapping_note": "Ca=0/1 ↔ R/S mapping is substrate-dependent; determined by majority vote in Step 4",
    "sources": [
        "Evans DA, Bartroli J, Shih TL. JACS 1981;103:2127",
        "Evans DA et al. JACS 1982;104:1737",
        "Zimmerman HE, Traxler MD. JACS 1957;79:1920",
        "Masamune S, Choy W. Aldrichimica Acta 1982;15:47",
    ],
    "confirmed": True,
}


def run(context: dict) -> dict:
    """Write SA convention confirmation to JSON."""
    out_dir: Path = context["output_dir"] / "literature"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sa_convention.json"
    with open(out_path, "w") as f:
        json.dump(SA_CONVENTION, f, indent=2)
    logger.info(f"Step 0: SA convention written to {out_path}")
    return context
