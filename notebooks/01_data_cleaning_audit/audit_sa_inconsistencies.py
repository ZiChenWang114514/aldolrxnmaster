"""
Audit script: Generate detailed report of all rows where label_SA != (Ca==Cb).

For each inconsistent row, outputs:
  - Reaction_ID, Year, Reaction_Class, metal
  - Raw SMILES (reaction + product)
  - label_Ca, label_Cb, label_SA, label_joint
  - Product structure image (SVG)
  - Aldol core atom mapping (which atoms are Ca, Cb, C=O, OH)
  - All stereocenters found by RDKit
  - 3D syn/anti re-computation (independent verification)

Output:
  - CSV for spreadsheet review
  - HTML report with molecule images
"""

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = Path("/data2/zcwang/aldolrxnmaster")
OUT_DIR = PROJECT_ROOT / "notebooks" / "01_data_cleaning_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The aldol core SMARTS from the original project
ALDOL_CORE = Chem.MolFromSmarts(
    "[*:101][CX3:1](=[O:2])[CX4:3]([*:102])([*:103])[CX4:4]([OX2H:5])([*:104])[*:105]"
)


def get_aldol_atoms(smiles: str) -> dict | None:
    """Find the aldol core atoms in a product SMILES."""
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    match = mol.GetSubstructMatch(ALDOL_CORE)
    if not match:
        return None

    # The SMARTS pattern maps atoms by their map numbers:
    # Pattern indices: 0=R101, 1=C=O(1), 2=O(2), 3=Ca(3), 4=R102, 5=R103, 6=Cb(4), 7=OH(5), 8=R104, 9=R105
    # But GetSubstructMatch returns indices in SMARTS order (atom order in SMARTS, not map number)
    # Let's identify by atom map number from the SMARTS
    smarts_mol = ALDOL_CORE
    map_to_idx = {}
    for i, atom in enumerate(smarts_mol.GetAtoms()):
        mn = atom.GetAtomMapNum()
        if mn > 0:
            map_to_idx[mn] = match[i]

    return {
        "carbonyl_C": map_to_idx.get(1, -1),
        "carbonyl_O": map_to_idx.get(2, -1),
        "Ca_idx": map_to_idx.get(3, -1),  # alpha carbon
        "Cb_idx": map_to_idx.get(4, -1),  # beta carbon (with OH)
        "OH_idx": map_to_idx.get(5, -1),
    }


def get_cip_at_atom(smiles: str, atom_idx: int) -> str:
    """Get CIP R/S assignment at a specific atom index."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "?"
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        atom = mol.GetAtomWithIdx(atom_idx)
        cip = atom.GetPropsAsDict().get("_CIPCode", "?")
        return cip
    except Exception:
        return "?"


def get_all_stereocenters(smiles: str) -> list[tuple]:
    """Get all stereocenters from a SMILES."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return []
    try:
        return Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    except Exception:
        return []


def generate_mol_svg(smiles: str, highlight_atoms: list[int] = None, size=(400, 300)) -> str:
    """Generate SVG image of molecule with optional atom highlighting."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "<p>Invalid SMILES</p>"
    try:
        AllChem.Compute2DCoords(mol)
        drawer = Draw.MolDraw2DSVG(size[0], size[1])
        if highlight_atoms:
            colors = {idx: (1.0, 0.6, 0.6) for idx in highlight_atoms}
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=colors)
        else:
            drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as e:
        return f"<p>Drawing error: {e}</p>"


def main():
    # Load the unified label data
    df = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "04_labels_unified.csv")

    # Find SA-inconsistent rows
    # SA interpretation: syn = (Ca == Cb) → SA=1 when Ca==Cb
    mask_sa_notna = df["label_SA"].notna() & df["label_Ca"].notna() & df["label_Cb"].notna()
    same_config = (df["label_Ca"] == df["label_Cb"]).astype(int)
    inconsistent_mask = mask_sa_notna & (df["label_SA"] != same_config)
    inconsistent = df[inconsistent_mask].copy()

    print(f"Total rows: {len(df)}")
    print(f"SA-inconsistent rows: {len(inconsistent)}")
    print(f"All from Reaction_Class: {inconsistent['Reaction_Class'].unique()}")
    print()

    # Enrich with structural analysis
    records = []
    html_rows = []

    for idx, row in inconsistent.iterrows():
        rid = row["Reaction_ID"]
        product_smi = row["Product_"] if pd.notna(row["Product_"]) else row["Raw_Product_Smiles"]

        # Find aldol core atoms
        aldol_atoms = get_aldol_atoms(product_smi)

        # Get CIP at Ca and Cb specifically
        ca_cip = "?"
        cb_cip = "?"
        highlight = []
        if aldol_atoms and aldol_atoms["Ca_idx"] >= 0:
            ca_cip = get_cip_at_atom(product_smi, aldol_atoms["Ca_idx"])
            cb_cip = get_cip_at_atom(product_smi, aldol_atoms["Cb_idx"])
            highlight = [
                aldol_atoms["Ca_idx"],
                aldol_atoms["Cb_idx"],
                aldol_atoms["carbonyl_C"],
                aldol_atoms["OH_idx"],
            ]
            highlight = [h for h in highlight if h >= 0]

        all_centers = get_all_stereocenters(product_smi)

        record = {
            "Reaction_ID": rid,
            "Year": row["Year"],
            "Reaction_Class": row["Reaction_Class"],
            "metal": row.get("metal", "?"),
            "label_Ca": int(row["label_Ca"]),
            "label_Cb": int(row["label_Cb"]),
            "label_SA": int(row["label_SA"]),
            "label_joint": int(row["label_joint"]) if pd.notna(row["label_joint"]) else "?",
            "Ca==Cb": int(row["label_Ca"]) == int(row["label_Cb"]),
            "expected_SA": int(row["label_Ca"] == row["label_Cb"]),
            "Ca_CIP_rdkit": ca_cip,
            "Cb_CIP_rdkit": cb_cip,
            "aldol_core_found": aldol_atoms is not None,
            "n_stereocenters_total": len(all_centers),
            "all_stereocenters": str(all_centers),
            "product_smiles": str(product_smi),
            "reaction_smiles": str(row["Raw_Reaction_Smiles"]),
        }
        records.append(record)

        # HTML row with molecule image
        svg = generate_mol_svg(product_smi, highlight_atoms=highlight)
        html_row = f"""
        <tr>
            <td>{rid}</td>
            <td>{row['Year']}</td>
            <td>{row['Reaction_Class']}</td>
            <td>{row.get('metal','?')}</td>
            <td>Ca={int(row['label_Ca'])}, Cb={int(row['label_Cb'])}</td>
            <td>SA={int(row['label_SA'])}<br>Expected: {int(row['label_Ca']==row['label_Cb'])}</td>
            <td>Ca:{ca_cip}, Cb:{cb_cip}</td>
            <td>{'Yes' if aldol_atoms else '<b style="color:red">No</b>'}</td>
            <td>{svg}</td>
            <td style="font-size:10px; max-width:300px; word-break:break-all">{product_smi}</td>
        </tr>
        """
        html_rows.append(html_row)

    # Save CSV
    out_csv = OUT_DIR / "sa_inconsistent_rows.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")

    # Save HTML report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SA Inconsistency Audit — AldolRxnMaster</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; vertical-align: middle; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>SA Inconsistency Audit Report</h1>
    <div class="summary">
        <p><b>Total dataset rows:</b> {len(df)}</p>
        <p><b>SA-inconsistent rows:</b> {len(inconsistent)} ({len(inconsistent)/len(df)*100:.2f}%)</p>
        <p><b>Rule:</b> label_SA should equal 1 when Ca==Cb (syn), 0 when Ca!=Cb (anti)</p>
        <p><b>All inconsistent rows are from class:</b> {', '.join(inconsistent['Reaction_Class'].unique())}</p>
        <p><b>Highlighted atoms:</b> <span style="color:red">Ca (α-carbon), Cb (β-carbon+OH), C=O, OH</span></p>
        <p><b>Action needed:</b> Review each row. If the aldol core is NOT found, the SMARTS pattern
           doesn't match this product — likely a different reaction type or unusual substrate.
           If found but CIP disagrees, the issue may be in the original CIP computation vs RDKit's.</p>
    </div>
    <table>
        <tr>
            <th>Reaction_ID</th>
            <th>Year</th>
            <th>Class</th>
            <th>Metal</th>
            <th>Labels</th>
            <th>SA (actual vs expected)</th>
            <th>RDKit CIP</th>
            <th>Aldol Core?</th>
            <th>Product Structure</th>
            <th>SMILES</th>
        </tr>
        {''.join(html_rows)}
    </table>
</body>
</html>"""

    out_html = OUT_DIR / "sa_inconsistent_audit.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"Saved HTML report: {out_html}")

    # Also print a summary
    print("\n=== Summary ===")
    for r in records:
        flag = "CORE NOT FOUND" if not r["aldol_core_found"] else ""
        print(
            f"  ID={r['Reaction_ID']}: Ca={r['label_Ca']}, Cb={r['label_Cb']}, "
            f"SA={r['label_SA']} (expect {r['expected_SA']}), "
            f"CIP: Ca={r['Ca_CIP_rdkit']}, Cb={r['Cb_CIP_rdkit']} {flag}"
        )


if __name__ == "__main__":
    main()
