from __future__ import annotations

import torch
from functools import lru_cache
from typing import Tuple, Any, Sequence, Optional


class Rotation:

    def __init__(self, rot_mats: Optional[torch.Tensor]):

        if rot_mats is None:
            raise ValueError('rot matrix must be specified')
        else:
            rot_mats = rot_mats.to(dtype=torch.float32)

        self._rot_mats = rot_mats

    # Magic methods
    def __getitem__(self, index):

        if type(index) != tuple:
            index = (index,)

        if self._rot_mats is not None:
            rot_mats = self._rot_mats[index + (slice(None), slice(None))]
            print("rot_mats in =====",rot_mats.shape)
            return Rotation(rot_mats=rot_mats)
        else:
            raise ValueError("rotation are None")
        
    def __mul__(self,
        right: torch.Tensor,
    ) -> Rotation:
        """
            Pointwise left multiplication of the transformation with a tensor.
            Can be used to e.g. mask the Rotation.

            Args:
                right:
                    The tensor multiplicand
            Returns:
                The product
        """
        if not(isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        new_rots = self._rot_mats * right.unsqueeze(-1).unsqueeze(-1) # [128,1,8,3,3] * [128, 14, 8] (3,3)
        return Rotation(new_rots)

    def get_rot_mat(self) -> torch.Tensor:
        """
        Return the underlying tensor rather than the Rotation object
        """
        return self._rot_mats


    def unsqueeze(self, dim: int) -> Rotation:

        if dim >= len(self.shape):
            raise ValueError("Invalid dimension for Rotation object")

        rot_mats = self._rot_mats.unsqueeze(dim if dim >=0 else dim -2)
        return Rotation(rot_mats=rot_mats)

    @property
    def shape(self) -> torch.Size:
        """
        Returns the virtual shape of the rotation object.
        If the Rotation was initialized with a [10, 3, 3]
        rotation matrix tensor, for example, the resulting shape would be
        [10].
        """
        s = self._rot_mats.shape[:-2]

        return s

    @staticmethod
    def identity(shape,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 requires_grad: bool = True) -> Rotation:

        rot_mats = torch.eye(3, dtype=dtype, device=device, requires_grad= requires_grad)
        rot_mats = rot_mats.view(*((1,) * len(shape)), 3, 3) # add empty dim of shape
        rot_mats = rot_mats.expand(*shape, -1, -1)
        rot_mats = rot_mats.contiguous()

        return rot_mats

class Rigid:
    """
        A class representing a rigid transformation. Little more than a wrapper
        around two objects: a Rotation object and a [*, 3] translation
        Designed to behave approximately like a single torch tensor with the
        shape of the shared batch dimensions of its component parts.
    """

    def __init__(self,
                 rots: Rotation,
                 trans: Optional[torch.Tensor],
                 ):
        if trans is None:
            batch_dims = rots.shape
            dtype = rots.get_rot_mat().dtype
            device = rots.get_rot_mat().device
            requires_grad = rots.get_rot_mat().requires_grad
            trans = identity_trans(batch_dims, dtype, device, requires_grad)


        self.rot = rots
        self.trans = trans

    @staticmethod # 不要求class被实例化，可以像 xx.py xx.identity 一样调用
    def identity(shape: Tuple[int],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 requires_grad: bool = True, # 应该是False吧，这里好像不用加梯度
                 ) -> Rigid:
        """
            Constructs an identity transformation.

            Args:
                shape:
                    The desired shape
                dtype:
                    The dtype of both internal tensors
                device:
                    The device of both internal tensors
                requires_grad:
                    Whether grad should be enabled for the internal tensors
            Returns:
                The identity transformation
        """

        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad),
            identity_trans(shape, dtype, device, requires_grad),
        )

    def __getitem__(self, index: Any) -> Rigid:

        if type(index) != tuple:
            index = (index,)

        return Rigid(self.rot[index], self.trans[index + (slice(None),)])
    
    def __mul__(self,
        right: torch.Tensor,
    ) -> Rigid:
        """
            Pointwise left multiplication of the transformation with a tensor.
            Can be used to e.g. mask the Rigid.

            Args:
                right:
                    The tensor multiplicand
            Returns:
                The product
        """
        if not(isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        new_rots = self.rot * right
        new_trans = self.trans * right[..., None]

        return Rigid(new_rots, new_trans)

    def unsqueeze(self, dim: int) -> Rigid:

        if dim >= len(self.shape):
            raise  ValueError("Invalid dimension for Rigid object")

        rot = self.rot.unsqueeze(dim)
        trans = self.trans.unsqueeze(dim if dim >=0 else dim -1)

        return Rigid(rot, trans)

    @property
    def shape(self) -> torch.Size:

        s = self.trans.shape[:-1]
        return s



def from_tensor_4x4(t: torch.Tensor) -> Rigid:
    """
        Constructs a transformation from a homogenous transformation
        tensor.

        Args:
            t: [*, 4, 4] homogenous transformation tensor
        Returns:
            T object with shape [*]
    """
    if t.shape[-2:] != (4, 4):
        raise ValueError("Incorrectly shaped input tensor")

    rots = Rotation(rot_mats=t[..., :3, :3],)
    trans = t[..., :3, 3]

    return Rigid(rots, trans)

@lru_cache(maxsize=None)
def identity_trans(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> torch.Tensor:
    trans = torch.zeros(
        (*batch_dims, 3),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )
    return trans

def Rigid_mult(rigid_1: Rigid,
               rigid_2: Rigid) -> Rigid:
    rot1 = rigid_1.rot.get_rot_mat()
    rot2 = rigid_2.rot.get_rot_mat()

    new_rot = rot_matmul(rot1, rot2)
    new_trans = rot_vec(rot1, rigid_2.trans)

    return  Rigid(Rotation(new_rot), new_trans)

def rigid_mul_vec(rigid: Rigid,
                  vec: torch.Tensor) -> torch.Tensor:

    rot_mat = rigid.rot.get_rot_mat()

    rotated = rot_vec(rot_mat, vec)

    return rotated + rigid.trans

def rot_matmul(
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """
        Performs matrix multiplication of two rotation matrix tensors. Written
        out by hand to avoid AMP downcasting.

        Args:
            a: [*, 3, 3] left multiplicand
            b: [*, 3, 3] right multiplicand
        Returns:
            The product ab
    """
    def row_mul(i):
        return torch.stack(
            [
                a[..., i, 0] * b[..., 0, 0]
                + a[..., i, 1] * b[..., 1, 0]
                + a[..., i, 2] * b[..., 2, 0],
                a[..., i, 0] * b[..., 0, 1]
                + a[..., i, 1] * b[..., 1, 1]
                + a[..., i, 2] * b[..., 2, 1],
                a[..., i, 0] * b[..., 0, 2]
                + a[..., i, 1] * b[..., 1, 2]
                + a[..., i, 2] * b[..., 2, 2],
            ],
            dim=-1,
        )

    return torch.stack(
        [
            row_mul(0),
            row_mul(1),
            row_mul(2),
        ],
        dim=-2
    )

def rot_vec(
    r: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
        Applies a rotation to a vector. Written out by hand to avoid transfer
        to avoid AMP downcasting.

        Args:
            r: [*, 3, 3] rotation matrices
            t: [*, 3] coordinate tensors
        Returns:
            [*, 3] rotated coordinates
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )

def cat(rigids: Sequence[Rigid],
        dim: int) -> Rigid:
    rot_mats = [r.rot.get_rot_mat() for r in rigids]
    rot_mats = torch.cat(rot_mats, dim=dim if dim >= 0 else dim - 2)
    rots = Rotation(rot_mats= rot_mats)

    trans = torch.cat([r.trans for r in rigids], dim=dim if dim >= 0 else dim - 1)

    return Rigid(rots, trans)

def map_rigid_fn(rigid: Rigid):

    rot_mat = rigid.rot.get_rot_mat()
    rot_mat = rot_mat.view(rot_mat.shape[:-2] + (9,))
    rot_mat = torch.stack(list(map(
        lambda x: torch.sum(x, dim=-1), torch.unbind(rot_mat, dim=-1)
    )), dim=-1)
    rot_mat = rot_mat.view(rot_mat.shape[:-1] + (3,3))

    new_trans = torch.stack(list(map(
        lambda x: torch.sum(x, dim=-1), torch.unbind(rigid.trans, dim=-1)
    )), dim=-1)

    return Rigid(Rotation(rot_mats=rot_mat), new_trans)

def get_gb_trans(bb_pos: torch.Tensor) -> Rigid: # [*,128,4,3]

    '''
    Get global transformation from given backbone position
    '''
    ex = bb_pos[..., 2, :] - bb_pos[..., 1, :] # [*,128,3]
    y_vec = bb_pos[..., 0, :] - bb_pos[..., 1, :] # [*,128,3]
    t = bb_pos[..., 1, :] # [*,128,3]
    
    print("ex====",ex.shape)
    print("y_vec====",y_vec.shape)
    print("t====",t.shape)
    print("torch.linalg.vector_norm(ex, dim=-1)", torch.linalg.vector_norm(ex, dim=-1).shape)
    # [*, N, 3]
    ex_norm = ex / torch.linalg.vector_norm(ex, dim=-1).unsqueeze(-1)
    print(ex_norm.shape)
    def dot(a, b):  # [*, N, 3]
        x1, y1, z1 = torch.unbind(b, dim=-1)
        x2, y2, z2 = torch.unbind(a, dim=-1)

        return x1 * x2 + y1 * y2 + z1 * z1

    ey = y_vec - dot(y_vec, ex_norm).unsqueeze(-1) * ex_norm
    ey_norm = ey / torch.linalg.norm(ey, dim=-1).unsqueeze(-1)

    ez_norm = torch.cross(ex_norm, ey_norm, dim=-1)

    '''
    m = torch.stack([ex_norm, ey_norm, ez_norm, t], dim=-1)
    m = torch.transpose(m, -2, -1)
    last_dim = torch.tensor([[0.0], [0.0], [0.0], [1.0]]).expand(6,12,4,1)

    m = torch.cat([m, last_dim], axis=-1)
    '''

    new_rot = torch.stack([ex_norm, ey_norm, ez_norm], dim=-1)
    return Rigid(Rotation(rot_mats=new_rot), t)