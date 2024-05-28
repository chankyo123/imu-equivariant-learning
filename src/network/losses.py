import torch
import numpy as np
from network.covariance_parametrization import DiagonalParam, CovarianceParam, PearsonParam, SinhParam

MIN_LOG_STD = np.log(1e-3)


"""
MSE loss between prediction and target, no logstdariance

input: 
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
output:
  loss: Nx3 vector of MSE loss on x,y,z
"""


def loss_mse(pred, targ):
    loss = (pred - targ).pow(2)
    return loss


"""
Log Likelihood loss, with logstdariance (only support diag logstd)

input:
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
  pred_logstd: Nx3 vector of log(sigma) on the diagonal entries
output:
  loss: Nx3 vector of likelihood loss on x,y,z

resulting pred_logstd meaning:
pred_logstd:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
"""


def loss_distribution_diag(pred, pred_logstd, targ):
    # min_value = -10.0  # Example minimum value
    # max_value = 10.0   # Example maximum value

    # # Clamp pred_logstd
    # clamped_pred_logstd = torch.clamp(pred_logstd, min=min_value, max=max_value)
    # print()
    pred_logstd = torch.maximum(pred_logstd, MIN_LOG_STD * torch.ones_like(pred_logstd))
    loss = ((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_logstd)) + pred_logstd
    return loss


"""
Log Likelihood loss, with logstdariance (support full logstd)
(NOTE: output is Nx1)

input:
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
  pred_logstd: Nxk logstdariance parametrization
output:
  loss: Nx1 vector of likelihood loss

resulting pred_logstd meaning:
DiagonalParam:
pred_logstd:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
PearsonParam:
pred_logstd (Nx6): u = [log(sigma_x) log(sigma_y) log(sigma_z)
                     rho_xy, rho_xz, rho_yz] (Pearson correlation coeff)
FunStuff
"""


def criterion_distribution(pred, covariance, targ, use_DiagonalParam):
    if use_DiagonalParam == 1: 
      loss = DiagonalParam.toMahalanobisDistance(
          targ, pred, covariance, clamp_covariance=False
      )
    elif use_DiagonalParam == 2 :
      loss = PearsonParam.toMahalanobisDistance(
      # loss = SinhParam.toMahalanobisDistance(
          targ, pred, covariance, clamp_covariance=False
      )
    elif use_DiagonalParam == 3 :
      loss = CovarianceParam.toMahalanobisDistance(
          targ, pred, covariance, clamp_covariance=False
      )
    return loss


"""
Select loss function based on epochs
all variables on gpu
output:
  loss: Nx3
"""
def get_loss(pred, pred_logstd, targ, epoch, body_frame_3regress = False):
    """
    if epoch < 10:
        loss = loss_mse(pred, targ)
    else:
        loss = loss_distribution_diag(pred, pred_logstd, targ)
    """
    use_DiagonalParam = 1 # diagonal : 1, pearson : 2, or entire cov : 3
    # if epoch < 10:
    if epoch < 30:
        pred_logstd = pred_logstd.detach()

    if body_frame_3regress: 
      # loss = criterion_distribution(pred, pred_logstd, targ, use_DiagonalParam)
      loss = loss_mse(pred, targ)
    else:
      # if epoch < 10 or epoch > 90:
      #   loss = loss_mse(pred, targ)
        
      # else: 
      #   loss = loss_distribution_diag(pred, pred_logstd, targ)
        
      loss = loss_distribution_diag(pred, pred_logstd, targ)
        

    return loss
