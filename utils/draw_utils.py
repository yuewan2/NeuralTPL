from utils.tpl_utils.generate_retro_templates import clear_mapnum
from utils.tpl_utils.test_mask_tpl import smi_tokenizer

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
try:
    from IPython.display import SVG, display
    from rdkit.Chem.Draw import IPythonConsole
except:
    pass


def clear_mapnum_rxn(rxn):
    c, ab = rxn.split('>>')
    c, ab = Chem.MolFromSmarts(c), Chem.MolFromSmarts(ab)
    clear_mapnum(c)
    clear_mapnum(ab)
    c, ab = Chem.MolToSmiles(c), Chem.MolToSmiles(ab)
    return '{}>>{}'.format(c, ab)


def draw_rxn(gt, size=(900, 300), highlight=False):
    rxn = AllChem.ReactionFromSmarts(gt)
    d = Draw.MolDraw2DSVG(size[0], size[1])
    colors = [(1, 0.6, 0.6), (0.4, 0.6, 1)]

    d.DrawReaction(rxn, highlightByReactant=highlight, highlightColorsReactants=colors)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    svg2 = svg.replace('svg:', '')
    svg3 = SVG(svg2)
    display(svg3)

# LATEST draw function:
def get_pair(atoms):
    atom_pairs = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            atom_pairs.append((atoms[i], atoms[j]))
    return atom_pairs


def get_bonds(mol, atoms):
    atom_pairs = get_pair(atoms)
    bond_list = []
    for ap in atom_pairs:
        bond = mol.GetBondBetweenAtoms(*ap)
        if bond is not None:
            bond_list.append(bond.GetIdx())
    return list(set(bond_list))


def draw_mols(smiles_list, smarts_list=None, noise_smarts_list=None, size=(500, 500)):
    import matplotlib
    color1 = matplotlib.colors.to_rgb('lightcoral')
    color2 = matplotlib.colors.to_rgb('cornflowerblue')  # cornflowerblue; darkseagreen
    mol_list, matched_atom_list, noise_matched_atom_list = [], [], []
    matched_bond_list, noise_matched_bond_list = [], []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        mol_list.append(mol)
        if smarts_list is not None:
            patt = Chem.MolFromSmarts(smarts_list[i])
            atoms_matched = mol.GetSubstructMatch(patt)
            bonds_matched = get_bonds(mol, atoms_matched)
            matched_atom_list.append(atoms_matched)
            matched_bond_list.append(bonds_matched)

        if noise_smarts_list is not None:
            patt = Chem.MolFromSmarts(noise_smarts_list[i])
            atoms_matched = mol.GetSubstructMatch(patt)
            bonds_matched = get_bonds(mol, atoms_matched)
            noise_matched_atom_list.append(atoms_matched)
            noise_matched_bond_list.append(bonds_matched)

    all_matched_atom_list, all_matched_bond_list = [], []
    all_matched_atom_color, all_matched_bond_color = [], []

    for i in range(len(matched_atom_list)):
        if len(noise_matched_atom_list):
            all_matched_atom_list.append(matched_atom_list[i] + noise_matched_atom_list[i])
            all_matched_bond_list.append(matched_bond_list[i] + noise_matched_bond_list[i])
            atom2color = {a: color1 for a in matched_atom_list[i]}
            atom2color.update({a: color2 for a in noise_matched_atom_list[i]})
            bond2color = {b: color1 for b in matched_bond_list[i]}
            bond2color.update({b: color2 for b in noise_matched_bond_list[i]})
        else:
            all_matched_atom_list.append(matched_atom_list[i])
            all_matched_bond_list.append(matched_bond_list[i])
            atom2color = {a: color1 for a in matched_atom_list[i]}
            bond2color = {b: color1 for b in matched_bond_list[i]}

        all_matched_atom_color.append(atom2color)
        all_matched_bond_color.append(bond2color)

    #     matched_atom_list[0] += noise_matched_atom_list[0]
    #     matched_atom_list[1] += noise_matched_atom_list[1]
    #     print(matched_atom_list[0])
    svg = Draw.MolsToGridImage(mol_list, subImgSize=size, useSVG=True,
                               highlightAtomLists=all_matched_atom_list,
                               highlightAtomColors=all_matched_atom_color,
                               highlightBondLists=all_matched_bond_list,
                               highlightBondColors=all_matched_bond_color)
    # svg = SVG(svg)
    display(svg)