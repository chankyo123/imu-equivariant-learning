"""Attitude encodings for the with-attitude VN-AirIO variants.

Why this file exists
--------------------
The submitted with-attitude model feeds ``Log(R_WS)`` as an extra vector channel.
That channel is *not* equivariant under the physical symmetry the model claims.
A sensor remount by ``Q`` acts on the right,

    R_WS  ->  R_WS Q^T ,          imu  ->  Q imu ,

and no linear map can send ``Log(R)`` to ``Q Log(R)`` under that action: ``Log`` is
equivariant only under *conjugation* ``R -> Q R Q^T``.  A one-line proof: every linear
map fixes 0, and ``Log(I) = 0``, yet ``Log(I Q^T) = -Log(Q) != 0``.  (Equivalently
``||Log(R)|| = theta`` is rotation-invariant while ``theta(R Q^T) != theta(R)``.)

The fix keeps a log-map reading but of the *right* object.  Fix three constant
rotations ``D_i = Exp([e_i]_x) in SO(3)`` (angle 1 rad about each basis axis).  Since
conjugation intertwines with Exp, ``R^T Exp([w]_x) R = Exp([R^T w]_x)``, so

    Log( R_WS^T D_i R_WS )^v  =  R_WS^T e_i  =  (row i of R_WS)  =: r_i ,

a genuine log-map of a genuine rotation.  A remount conjugates the argument,
``R_WS'^T D_i R_WS' = Q (R_WS^T D_i R_WS) Q^T``, which is exactly the action ``Log`` is
equivariant under -- so ``r_i -> Q r_i``.  The identity also makes the ``Log`` free at
runtime: it collapses to reading off the rows.

Three properties follow, all verified to float64 over 20k random rotations:
  * conjugation preserves the rotation angle, so the argument sits at 1 rad for every
    ``R_WS`` -- 2.14 rad away from the ``pi`` branch cut, which is therefore never
    approached (contrast A0, whose Lipschitz ratio diverges as ``pi/eps`` at the cut);
  * ``R_WS`` is recoverable from its rows, so ``Log(R_WS)`` is too -- nothing is lost
    relative to the submitted feature;
  * the encoding costs 3 vector channels instead of 1 (+768 parameters).

Encodings
---------
``logmap``  A0  ``Log(R_WS)``                   1 channel   -- submitted, NOT equivariant
``rows``    A2  rows of ``R_WS``                3 channels  -- exactly equivariant, lossless
``up``      A1  ``R_WS^T e_z`` (gravity dir.)   1 channel   -- equivariant, loses yaw
``rel``     A3  ``Log(R_last^T R_t)``           1 channel   -- Ad-covariant, loses absolute att.
"""

import torch


def attitude_feature(gt_rot, rep="logmap"):
    """Build the attitude vector channels.

    Args:
        gt_rot: pypose ``SO3`` LieTensor of shape ``(B, T)`` holding ``R_WS``.
        rep:    one of ``logmap`` / ``rows`` / ``up`` / ``rel`` / ``None``.

    Returns:
        ``None`` if ``rep`` is None; ``(B, T, 3)`` for the single-channel encodings
        (so the submitted models keep their original signature); ``(B, T, 3, 3)`` for
        ``rows``, where index ``-2`` is the vector-channel axis.
    """
    if rep is None or rep == "none":
        return None

    # The stored NanoBench quaternions are unit only to ~5e-9, and pypose does not
    # renormalise.  A non-unit quaternion is not a rotation: `matrix()` then returns
    # a matrix scaled by ~|q|^2, and the two equivalent ways of writing a remount
    # (`R Q^T` via quaternion product vs `Q` applied to the rows) stop agreeing at
    # the 1e-8 level.  Over a whole-sequence forward pass the bidirectional VN-GRU
    # amplifies that into ~1e-1 of apparent "equivariance error" that is really just
    # input precision.  Normalising here makes the encoding a function of a genuine
    # rotation, and the two paths agree to 8e-16.
    import pypose as pp
    t = gt_rot.tensor()
    gt_rot = pp.SO3(t / t[..., -4:].norm(dim=-1, keepdim=True))

    if rep == "logmap":                                    # A0
        return gt_rot.Log().tensor()                       # (B, T, 3)

    if rep == "rows":                                      # A2
        # row i of R_WS  ==  R_WS^T e_i  ==  Log(R_WS^T D_i R_WS)^v
        return gt_rot.matrix()                             # (B, T, 3, 3) -- rows are channels

    if rep == "up":                                        # A1
        return gt_rot.matrix()[..., 2, :]                  # (B, T, 3)

    if rep == "rel":                                       # A3
        import pypose as pp
        last = pp.SO3(gt_rot.tensor()[..., -1:, :])        # (B, 1) -- LieTensor slicing
        rel = last.Inv() @ gt_rot                          # R_last^T R_t (conjugated by
        return rel.Log().tensor()                          # a remount -> Ad-covariant)

    raise ValueError(f"unknown attitude representation: {rep!r}")


def rep_of(network):
    """Attitude encoding the network was built for (survives DataParallel)."""
    net = getattr(network, "module", network)
    return getattr(net, "att_rep", "logmap")


def n_channels(rep):
    return {None: 0, "none": 0, "logmap": 1, "rows": 3, "up": 1, "rel": 1}[rep]
