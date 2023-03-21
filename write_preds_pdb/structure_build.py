import torch
import torch.nn as nn
import numpy as np
from constant import (restype_rigid_group_default_frame,
                      restype_atom14_to_rigid_group,
                      restype_atom14_mask,
                      restype_atom14_rigid_group_positions,
                      restype_atom14_to_atom37,
                      restype_atom37_mask)
import geometry
import protein

def rotate_sidechain(aatype: torch.Tensor,
                    restype_idx:torch.Tensor,
                    angles: torch.Tensor) -> geometry.Rigid:

    # [21, 8, 4, 4]
    default_frame = torch.tensor(restype_rigid_group_default_frame,
                              dtype=angles.dtype,
                              device=angles.device,
                              requires_grad=False)
    # [*, N, 8, 4, 4]
    res_default_frame = default_frame[restype_idx, ...]

    # [*, N, 8] Rigid
    default_r = geometry.from_tensor_4x4(res_default_frame)

    sin_angles = angles[..., 0] # [*,N,4]
    cos_angles = angles[..., 1]

    # [*,N,4] + [*,N,4] == [*,N,8]
    sin_angles = torch.cat([torch.zeros(*aatype.shape, 4), sin_angles],dim=-1)
    cos_angles = torch.cat([torch.ones(*aatype.shape, 4), cos_angles],dim=-1)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.
    all_rots = angles.new_zeros(default_r.rot.get_rot_mat().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = cos_angles
    all_rots[..., 1, 2] = -sin_angles
    all_rots[..., 2, 1] = sin_angles
    all_rots[..., 2, 2] = cos_angles

    all_rots = geometry.Rigid(geometry.Rotation(rot_mats = all_rots), None)

    all_frames = geometry.Rigid_mult(default_r,all_rots)

    # Rigid
    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = geometry.Rigid_mult(chi1_frame_to_bb, chi2_frame_to_frame)
    chi3_frame_to_bb = geometry.Rigid_mult(chi2_frame_to_bb, chi3_frame_to_frame)
    chi4_frame_to_bb = geometry.Rigid_mult(chi3_frame_to_bb, chi4_frame_to_frame)

    all_frames_to_bb = geometry.cat(
        [all_frames[..., :5],
        chi2_frame_to_bb.unsqueeze(-1),
        chi3_frame_to_bb.unsqueeze(-1),
        chi4_frame_to_bb.unsqueeze(-1),],
        dim=-1,
    )

    return all_frames_to_bb

def frame_to_pos(frames, aatype_idx):

    # [21 , 14]
    group_index = torch.tensor(restype_atom14_to_rigid_group)

    # [21 , 14] idx [*, N] -> [*, N, 14]
    group_mask = group_index[aatype_idx, ...]
    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(group_mask, num_classes = frames.shape[-1])

    # [*, N, 14, 8] Rigid
    map_atoms_to_global = frames[..., None, :] * group_mask

    map_atoms_to_global = geometry.map_rigid_fn(map_atoms_to_global)

    # [21 , 14]
    atom_mask = torch.tensor(restype_atom14_mask)
    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype_idx, ...].unsqueeze(-1)

    # [21, 14, 3]
    default_pos = torch.tensor(restype_atom14_rigid_group_positions)
    # [*, N, 14, 3]
    default_pos = default_pos[aatype_idx, ...]

    pred_pos = geometry.rigid_mul_vec(map_atoms_to_global, default_pos)
    pred_pos = pred_pos * atom_mask

    return pred_pos

def batch_gather(data,  # [N, 14, 3]
                 indexing): # [N,37]
    ranges = []

    N = data.shape[-3]
    r = torch.arange(N)
    r = r.view(-1,1)
    ranges.append(r)

    remaining_dims = [slice(None) for _ in range(2)]
    remaining_dims[-2] = indexing

    ranges.extend(remaining_dims)# [Tensor(N,1), Tensor(N,37), slice(None)]
    return data[ranges] # [N, 37, 3]

def atom14_to_atom37(atom14, aa_idx):
    residx_atom37_to_14 = restype_atom14_to_atom37[aa_idx]
    # [N, 37]
    atom37_mask = restype_atom37_mask[aa_idx]

    # [N, 37, 3]
    atom37 = batch_gather(atom14[-1], residx_atom37_to_14)
    atom37 = atom37 * atom37_mask[...,None]

    return atom37

def torsion_to_position(aatype_idx: torch.Tensor, # [*, N]
                        backbone_position: torch.Tensor, # [*, N, 4, 3] (N, CA, C, O)
                        angles: torch.Tensor, # [*, N, 4, 2] (X1, X2, X3, X4) (sin, cos)
                        ): # -> [*, N, 14, 3]
    """Compute Side Chain Atom position using the predicted torsion
    angle and the fixed backbone coordinates.

    Args:
        aatype_idx: aatype for each residue
        backbone_position: backbone coordinate for each residue
        angles: torsion angles for each residue

    return:
        all atom position [N, X] X are # number of atoms (14?
    """

    # side chain frames [*, N, 8] Rigid
    angles_sin_cos = torch.stack(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )
    sc_to_bb = rotate_sidechain(aatype_idx, angles_sin_cos)

    # [*, N] Rigid

    bb_to_gb = geometry.get_gb_trans(backbone_position)

    ''''
    bb_to_gb = torch.tensor(
                            [make_rigid_trans(
                            ex = res[2] - res[1], # C-CA,
                            y_vec = res[0] - res[1], # N-CA
                            t = res[1])  for res in backbone_position]
    ) # [N, 4, 4]
    '''

    all_frames_to_global = geometry.Rigid_mult(bb_to_gb[..., None], sc_to_bb)

    # [*, N, 14, 3]
    all_pos = frame_to_pos(all_frames_to_global, aatype_idx)

    # [*, N, 37, 3]
    final_pos = atom14_to_atom37(all_pos, aatype_idx)

    return final_pos



    

#############################################################
#result = {}
#features = {}

#features["final_atom_mask"] = restype_atom37_mask[features["aatype_idx"]]
# [*,N,14,3]
#result["final_atom_positions"] = torsion_to_position(features["aatype_idx"], 
#                                                     features["backbone_position"],
#                                                     result["angles"])# [*,N...]

#resulted_protein = protein.Protein(
  #                          aatype=features["aatype_idx"], # [*,N]
  #                          atom_positions=result["final_atom_positions"],
  #                          atom_mask=features["final_atom_mask"],
  #                          residue_index=features["residue_index"] + 1,
 #                           b_factors=np.zeros_like(features["final_atom_mask"]))

#pdb_str = protein.to_pdb(resulted_protein) 
#produced_pdb_file_path = '' # given a pdb_path
#with open(produced_pdb_file_path, 'w') as fp:
#    fp.write(pdb_str)
    

def write_preds_pdb_file(dataset, sampled_dfs, out_path):
    
    final_atom_mask = restype_atom37_mask[dataset[0]["seq"]]
    final_atom_positions = torsion_to_position(dataset[0]["seq"], 
                                                     dataset[0]["coords"],
                                                     sampled_dfs[0])
    chain_len = len([0]["seq"])
    index = np.arange(1,chain_len+1)
    resulted_protein = protein.Protein(
                            aatype=dataset[0]["seq"], # [*,N]
                            atom_positions=final_atom_positions,
                            atom_mask=final_atom_mask,
                            residue_index=index, #0,1,2,3,4 range_chain
                            b_factors=np.zeros_like(final_atom_mask))
    
    pdb_str = protein.to_pdb(resulted_protein) 
    with open(out_path+'generate_0', 'w') as fp:
         fp.write(pdb_str)
    