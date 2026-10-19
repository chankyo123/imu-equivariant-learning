from .code import *
from .vn_codenet import (
    VNCodeNetMotion, VNCodeNetMotionwithRot,
    VNCodeNetMotionSmall, VNCodeNetMotionwithRotSmall,
    VNCodeNetMotionSpectralSmall, VNCodeNetMotionA2Small,
    VNCodeNetMotionA0SpecSmall, VNCodeNetMotionA1Small, VNCodeNetMotionA3Small,
    VNCodeNetMotionA2DiagSmall,
)
from .attitude import attitude_feature, rep_of, n_channels

net_dict = {
    'codenetmotion':CodeNetMotion,
    'codewithrot':CodeNetMotionwithRot,
    'vn_codenetmotion': VNCodeNetMotion,
    'vn_codewithrot': VNCodeNetMotionwithRot,
    'vn_codenetmotion_small': VNCodeNetMotionSmall,
    'vn_codewithrot_small': VNCodeNetMotionwithRotSmall,
    # equivariant attitude encodings + spectral covariance
    'vn_noatt_spec_small': VNCodeNetMotionSpectralSmall,
    'vn_a2_spec_small': VNCodeNetMotionA2Small,
    'vn_a0_spec_small': VNCodeNetMotionA0SpecSmall,
    'vn_a1_spec_small': VNCodeNetMotionA1Small,
    'vn_a3_spec_small': VNCodeNetMotionA3Small,
    'vn_a2_diag_small': VNCodeNetMotionA2DiagSmall,
}
