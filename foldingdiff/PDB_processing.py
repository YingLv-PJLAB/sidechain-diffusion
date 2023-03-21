from Bio import PDB
import math
import numpy as np
import pandas as pd

ANGLES = ["X1", "X2", "X3","X4"]
chi_angles_atoms = {
    "ALA": [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    "ARG": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "NE"],
        ["CG", "CD", "NE", "CZ"],
    ],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLU": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "OE1"],
    ],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "CD"],
        ["CB", "CG", "CD", "CE"],
        ["CG", "CD", "CE", "NZ"],
    ],
    "MET": [
        ["N", "CA", "CB", "CG"],
        ["CA", "CB", "CG", "SD"],
        ["CB", "CG", "SD", "CE"],
    ],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}
restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}
bb_atoms = ['N', 'CA', 'C', 'O']

def get_torsion_seq(pdb_path):
    torsion_list = []
    seq = []
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('name', pdb_path)
    model = structure[0]
    chain = model.child_list[0]
    X = []

    for res_idx, res in enumerate(chain):

        chi_list = [0] * 4
        # Skip hetero atoms
        if res.id[0] != " ":
            continue

        res_name = res.resname
        seq.append(restype_3to1[res_name])
        res_torsion_atom_list = chi_angles_atoms[res_name]
        X.append([res[a].get_coord() for a in bb_atoms])
        
        for i, torsion_atoms in enumerate(res_torsion_atom_list):
            vec_atoms_coord = [res[a].get_vector() for a in torsion_atoms]
            angle = PDB.calc_dihedral(*vec_atoms_coord)
            chi_list[i] = angle

        torsion_list.append(chi_list)
         
        
    torsion_list = np.array(torsion_list)
    X = np.array(X)
    
    #=========
    seq = np.array(seq, dtype=np.str)
    X1 = torsion_list[:,0]
    X2 = torsion_list[:,1]
    X3 = torsion_list[:,2]
    X4 = torsion_list[:,3]
    calc_angles = {"X1": X1, "X2": X2, "X3": X3, "X4": X4}
    angle_list = pd.DataFrame({k: calc_angles[k].squeeze() for k in ANGLES})
    dict_struct = {'angles':angle_list,'coords': X, 'seq': seq, 'fname':pdb_path}
    return dict_struct

#t = get_torsion_seq('./data/1CRN.pdb')
#l = len(t1)
#l2 = len(t['seq'])
#seq= "".join(t["seq"])
#print(seq)  chi_1