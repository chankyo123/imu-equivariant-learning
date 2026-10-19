import numpy as np
import torch
from network.covariance_parametrization import DiagonalParam
from utils.logging import logging
from network.model_factory import get_model
import copy


class MeasSourceTorchScript:
    """ Loading a torchscript has the advantage that we do not need to reconstruct the original network class to
        load the weights, the network structure is contained into the torchscript file.
    """
    def __init__(self, model_path, arch, net_config, force_cpu=False,
                 secondary_model_path=None, secondary_arch=None, hybrid_mode=None):
        # load trained network model
        logging.info("Loding {}...".format(model_path))
        if not torch.cuda.is_available() or force_cpu:
            torch.init_num_threads()
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass  # Already set on prior sequence in batch mode
            self.device = torch.device("cpu")
            # self.net = torch.jit.load(model_path, map_location="cpu")

        else:
            self.device = torch.device("cuda:0")
            # NOTE TLIO baseline model won't work on GPU unless we ass map_location
            # https://github.com/pytorch/pytorch/issues/78207
            # self.net = torch.jit.load(model_path, map_location=self.device)#torch.jit.load(model_path, map_location=self.device)
        # Load TorchScript-traced model (model_path is the model_torchscript.pt produced by convert_model_to_torchscript.py)
        self.net = torch.jit.load(model_path, map_location=self.device)
        self.net.to(self.device)
        self.net.eval()
        logging.info("Model {} loaded to device {}.".format(model_path, self.device))

        # Hybrid evaluation: optionally load a secondary model and combine its (mean, cov) with the primary
        self.secondary_net = None
        self.secondary_arch = secondary_arch
        self.hybrid_mode = hybrid_mode  # 'mean_from_secondary' (B) or 'cov_from_secondary' (A) or None
        self._in_hybrid_call = False
        if secondary_model_path is not None:
            self.secondary_net = torch.jit.load(secondary_model_path, map_location=self.device).to(self.device).eval()
            logging.info(f"Secondary model loaded: {secondary_model_path} (arch={secondary_arch}, hybrid_mode={hybrid_mode})")
    
    def preprocess_for_eq_transformer(self, imu0, pe_ts):
        ## read the positional encoded timestamps
        scalar = pe_ts
        ## read the imu data
        feat = imu0
        feat = feat.permute(0,2,1)
        ## separate the vectors and scalars from features - gyro, accel
        gyro = torch.cat((feat[..., 0].unsqueeze(-1), feat[..., 1].unsqueeze(-1)), dim=-1).unsqueeze(-2)
        accel = torch.cat((feat[..., 3].unsqueeze(-1), feat[..., 4].unsqueeze(-1)), dim=-1).unsqueeze(-2)
        vector = torch.cat((accel, gyro), dim=-2)
        scalar = torch.cat((scalar, feat[..., 2].unsqueeze(-1), feat[..., -1].unsqueeze(-1), torch.norm(accel, dim=-1), torch.norm(gyro, dim=-1), torch.einsum('... a b, ... c b -> ... c', accel, gyro)), dim=-1)
        return vector.permute(0,1,3,2), scalar  

    def postprocess_eq_transformer(self,out_vec, out_sca):
        return torch.cat((out_vec.squeeze(-1), out_sca[..., 0].unsqueeze(-1)), dim=-1), out_sca[..., 1:]

    def postprocess_eq_transformer_3scalars(self,out_vec, out_sca):
        return torch.cat((out_vec.squeeze(-1), out_sca[..., 0].unsqueeze(-1)), dim=-1), torch.cat((out_sca[..., 1].unsqueeze(-1), out_sca[..., 1].unsqueeze(-1), out_sca[..., 2].unsqueeze(-1)), dim=-1)

    def postprocess_eq_transformer_6v_6s(self,out_vec, out_sca):
        cov = torch.cat((out_vec[..., 1:], out_sca[..., 1:].unsqueeze(-2)), dim=-2)
        mu = cov.mean(-1)
        cov = (torch.matmul((cov - mu.unsqueeze(-1)), torch.transpose(cov - mu.unsqueeze(-1), 1, 2))/(cov.shape[-1]-1))
        print('cov:', cov[0])
        return torch.cat((out_vec[..., 0].squeeze(-1), out_sca[..., 0].unsqueeze(-1)), dim=-1), cov

    def preprocess_wo_t(self,feat):
        ## read the imu data
        feat = feat.permute(0,2,1)
        ## separate the vectors and scalars from features - gyro, accel
        gyro = torch.cat((feat[..., 0].unsqueeze(-1), feat[..., 1].unsqueeze(-1)), dim=-1).unsqueeze(-2)
        accel = torch.cat((feat[..., 3].unsqueeze(-1), feat[..., 4].unsqueeze(-1)), dim=-1).unsqueeze(-2)
        vector = torch.cat((accel, gyro), dim=-2)
        scalar = torch.cat((feat[..., 2].unsqueeze(-1), feat[..., -1].unsqueeze(-1), torch.norm(accel, dim=-1), torch.norm(gyro, dim=-1), torch.einsum('... a b, ... c b -> ... c', accel, gyro)), dim=-1)
        return vector.permute(0,1,3,2), scalar 

    def preprocess_wo_t_tlio_frame(self, feat):
        ## read the imu data
        feat = feat.permute(0,2,1)
        ## separate the vectors and scalars from features - gyro, accel
        gyro = torch.cat((feat[..., 0].unsqueeze(-1), feat[..., 1].unsqueeze(-1)), dim=-1).unsqueeze(-2)
        accel = torch.cat((feat[..., 3].unsqueeze(-1), feat[..., 4].unsqueeze(-1)), dim=-1).unsqueeze(-2)
        vector = torch.cat((accel, gyro), dim=-2)
        scalar = torch.cat((feat[..., 2].unsqueeze(-1), feat[..., -1].unsqueeze(-1), torch.norm(accel, dim=-1), torch.norm(gyro, dim=-1), torch.einsum('... a b, ... c b -> ... c', accel, gyro)), dim=-1)
        return vector.permute(0,1,3,2), scalar, torch.cat((feat[..., 2].unsqueeze(-1), feat[..., -1].unsqueeze(-1)), dim=-1)

    def postprocess_tlio_frame_2scalars(self,pred, cov):
        return pred, torch.cat((cov[..., 0].unsqueeze(-1), cov[..., 0].unsqueeze(-1), cov[..., 1].unsqueeze(-1)), dim=-1)

    def preprocess_tlio_w_t(self, feat, pe_ts): #9 extra values
        return torch.cat((feat.permute(0,2,1),pe_ts), dim=-1).permute(0,2,1)
    
    def preprocess_o2(self,gyro, accel):
        ## gyro and accel in format N x 3
        v1 = np.zeros((*gyro.shape[:-1],3))
        v2 = np.zeros((*gyro.shape[:-1],3))
        mask = (np.linalg.norm(gyro[...,:-1], axis=-1) == 0).astype(np.int32)
        R = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        gyro_flip = copy.deepcopy(gyro@R.T)
        gyro_flip[...,-1] = 0
        v1[mask==0] = np.cross(gyro[mask==0], gyro_flip[mask==0])
        v2[mask==0] = np.cross(gyro[mask==0],v1[mask==0])
        x = np.zeros_like(gyro)
        x[...,-2] = 1
        v1[mask==1] = np.cross(x[mask==1], gyro[mask==1])
        v2[mask==1] = np.cross(gyro[mask==1], v1[mask==1])
        gyro_norm = np.linalg.norm(gyro,axis=-1,keepdims=True)
        v1 = v1 * gyro_norm /np.clip(np.linalg.norm(v1, axis=-1,keepdims=True),1e-7,1e13)
        v2 = v2 * gyro_norm/np.clip(np.linalg.norm(v2,axis=-1,keepdims=True),1e-7,1e13)

        accel = torch.from_numpy(accel).float().to(self.device)
        v1 = torch.from_numpy(v1).float().to(self.device)
        v2 = torch.from_numpy(v2).float().to(self.device)
        a = accel[...,:2].unsqueeze(-2)
        v1_xy = v1[...,:2].unsqueeze(-2)
        v2_xy = v2[...,:2].unsqueeze(-2)
        scalar = torch.cat((accel[...,-1].unsqueeze(-1), v1[...,-1].unsqueeze(-1), v2[...,-1].unsqueeze(-1),torch.norm(a,dim=-1),torch.norm(v1_xy,dim=-1),torch.norm(v2_xy,dim=-1),(a@v1_xy.permute(0,2,1)).squeeze(-1),(v1_xy@v2_xy.permute(0,2,1)).squeeze(-1),(a@v2_xy.permute(0,2,1)).squeeze(-1)),dim=-1)
        return torch.cat((a,v1_xy,v2_xy),dim=-2).permute(0,2,1).unsqueeze(0), scalar.unsqueeze(0), torch.cat((accel[...,-1].unsqueeze(-1), v1[...,-1].unsqueeze(-1), v2[...,-1].unsqueeze(-1)), dim=-1).unsqueeze(0)

    def preprocess_so3(self, gyro, accel):
        """SO(3) preprocessing for Eq_Motion_Model_SO3_fullCov.
        Inputs gyro/accel in numpy (N x 3). Returns torch tensors:
          vector: (1, N, 3, 3)   -- vec_dim=3 at -2, 3 features at -1
          scalar: (1, N, 6)      -- 3 norms + 3 dot products (SO(3)-invariant)
        Cross-product features built the same way as preprocess_o2; the only difference
        is that we keep all three components instead of splitting xy/z.
        """
        v1 = np.zeros((*gyro.shape[:-1], 3))
        v2 = np.zeros((*gyro.shape[:-1], 3))
        mask = (np.linalg.norm(gyro[..., :-1], axis=-1) == 0).astype(np.int32)
        Rflip = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        gyro_flip = copy.deepcopy(gyro @ Rflip.T)
        gyro_flip[..., -1] = 0
        v1[mask == 0] = np.cross(gyro[mask == 0], gyro_flip[mask == 0])
        v2[mask == 0] = np.cross(gyro[mask == 0], v1[mask == 0])
        x = np.zeros_like(gyro); x[..., -2] = 1
        v1[mask == 1] = np.cross(x[mask == 1], gyro[mask == 1])
        v2[mask == 1] = np.cross(gyro[mask == 1], v1[mask == 1])
        gyro_norm = np.linalg.norm(gyro, axis=-1, keepdims=True)
        v1 = v1 * gyro_norm / np.clip(np.linalg.norm(v1, axis=-1, keepdims=True), 1e-7, 1e13)
        v2 = v2 * gyro_norm / np.clip(np.linalg.norm(v2, axis=-1, keepdims=True), 1e-7, 1e13)

        a_t  = torch.from_numpy(accel).float().to(self.device)
        v1_t = torch.from_numpy(v1).float().to(self.device)
        v2_t = torch.from_numpy(v2).float().to(self.device)
        # invariant scalars
        na  = a_t.norm(dim=-1, keepdim=True)
        nv1 = v1_t.norm(dim=-1, keepdim=True)
        nv2 = v2_t.norm(dim=-1, keepdim=True)
        av1  = (a_t  * v1_t).sum(-1, keepdim=True)
        v1v2 = (v1_t * v2_t).sum(-1, keepdim=True)
        av2  = (a_t  * v2_t).sum(-1, keepdim=True)
        scalar = torch.cat([na, nv1, nv2, av1, v1v2, av2], dim=-1)            # (N, 6)
        vector = torch.stack([a_t, v1_t, v2_t], dim=-1)                        # (N, 3, 3)
        return vector.unsqueeze(0), scalar.unsqueeze(0)                        # (1, N, 3, 3), (1, N, 6)



    def get_displacement_measurement(self, net_gyr_w, net_acc_w, pe_ts, arch_type, clip_small_disp=False):
        # Hybrid wrapper: run primary then secondary, combine in world frame. Existing single-model logic untouched below.
        if self.secondary_net is not None and not self._in_hybrid_call:
            self._in_hybrid_call = True
            try:
                # Primary (uses self.net, arch_type passed in)
                meas_primary, cov_primary = self.get_displacement_measurement(
                    net_gyr_w, net_acc_w, pe_ts, arch_type, clip_small_disp=False
                )
                # Secondary (swap self.net temporarily)
                saved_net = self.net
                self.net = self.secondary_net
                try:
                    meas_secondary, cov_secondary = self.get_displacement_measurement(
                        net_gyr_w, net_acc_w, pe_ts, self.secondary_arch, clip_small_disp=False
                    )
                finally:
                    self.net = saved_net
                # Combine
                if self.hybrid_mode == 'mean_from_secondary':
                    meas, meas_cov = meas_secondary, cov_primary
                elif self.hybrid_mode == 'cov_from_secondary':
                    meas, meas_cov = meas_primary, cov_secondary
                else:
                    raise ValueError(f"Unknown hybrid_mode: {self.hybrid_mode}")
                # Apply clip on combined output (same as below)
                if clip_small_disp and np.linalg.norm(meas) < 0.001:
                    meas = 0 * meas
                return meas, meas_cov
            finally:
                self._in_hybrid_call = False

        with torch.no_grad():
            if arch_type == 'rnin_vio_model_lstm':
                net_acc_w = net_acc_w - np.array([0.0,0.0,9.805]).reshape((1,-1))
            features = np.concatenate([net_gyr_w, net_acc_w], axis=1)  # N x 6
            features_t = torch.unsqueeze(
                torch.from_numpy(features.T).float().to(self.device), 0
            )  # 1 x 6 x N
            pe_ts = torch.unsqueeze(torch.from_numpy(pe_ts).float().to(self.device), 0)
           
            # print('features shape:',features.shape)
            # print('features_t shape:', features_t.shape)

            if arch_type == 'eq_transformer':
                vector, scalar = self.preprocess_for_eq_transformer(features_t, pe_ts)
                out_vec, out_sca = self.net(vector.float(), scalar.float())
                outputs = self.postprocess_eq_transformer(out_vec, out_sca)
            elif 'resnet_2scalars_tlio_frame' in arch_type:
                frame,out_vec, out_sca = self.net(features_t)
                pred, pred_cov = self.postprocess_tlio_frame_2scalars(out_vec, out_sca)
                outputs = (pred, pred_cov)
            elif 'resnet_fullCov_tlio_frame' in arch_type:
                frame,out_vec, out_sca = self.net(features_t)
                outputs = (out_vec, out_sca)
            elif  '_3scalars' in arch_type and 'wo_t' not in arch_type:
                vector, scalar = self.preprocess_for_eq_transformer(features_t, pe_ts)
                out_vec, out_sca = self.net(vector.float(), scalar.float())
                outputs = self.postprocess_eq_transformer_3scalars(out_vec, out_sca)
            elif arch_type == 'rnin_vio_model_lstm':
                features_t = features_t.unsqueeze(dim=1)
                ## rnin is gravity compensated
                netargs = [features_t]
                pred, pred_cov = self.net(*netargs)
                outputs = (pred.squeeze(dim=1), pred_cov.squeeze(dim=1))
            elif '_wo_t_3scalars' in arch_type:
                vector, scalar = self.preprocess_wo_t(features_t)
                out_vec, out_sca = self.net(vector.float(), scalar.float())
                pred, pred_cov = self.postprocess_eq_transformer_3scalars(out_vec, out_sca)
                outputs = (pred.squeeze(dim=1), pred_cov.squeeze(dim=1))
            elif 'o2_frame_2scalars' in arch_type:
                frame,out_vec, out_sca = self.net(*self.preprocess_o2(net_gyr_w, net_acc_w))
                pred, pred_cov = self.postprocess_tlio_frame_2scalars(out_vec, out_sca)
                outputs = (pred, pred_cov)
            elif 'so3_frame' in arch_type:
                # SO(3)-equivariant extension. Forward returns (frame_3x3, disp_canonical, cov_canonical_diag_logstd)
                vector, scalar = self.preprocess_so3(net_gyr_w, net_acc_w)
                frame, pred, pred_cov = self.net(vector, scalar)
                outputs = (pred, pred_cov)
            elif 'o2_frame' in arch_type:
                vector,scalar,orig_sca = self.preprocess_o2(net_gyr_w, net_acc_w)
                frame,pred,pred_cov = self.net(vector,scalar,orig_sca)
                outputs = (pred, pred_cov)
            elif '_wo_t_tlio_frame_2scalars' in arch_type:
                vector, scalar, orig_sca = self.preprocess_wo_t_tlio_frame(features_t)
                frame,out_vec, out_sca = self.net(vector.float(), scalar.float(), orig_sca.float())
                pred, pred_cov = self.postprocess_tlio_frame_2scalars(out_vec, out_sca)
                outputs = (pred, pred_cov)
            elif 'tlio_frame_fullCov' in arch_type or '_PearsonCov' in arch_type:
                vector, scalar, orig_sca = self.preprocess_wo_t_tlio_frame(features_t)
                frame,pred, pred_cov = self.net(vector.float(), scalar.float(), orig_sca.float())
                outputs = (pred, pred_cov)
            # elif 'o2_frame' in arch_type: ## see more carefully how to add it
            #     frame,pred,pred_cov = self.net(self.preprocess_o2(features_t))
            elif arch_type == 'resnet_w_t':
                feat = self.preprocess_tlio_w_t(features_t, pe_ts)
                outputs = self.net(feat.float())
            else:
                netargs = [features_t]
                outputs = self.net(*netargs)


            if type(outputs) == tuple:  # Legacy
                meas, meas_cov = outputs
            elif type(outputs) == dict:  # New output format
                meas, meas_cov = outputs["pred"], outputs["pred_log_std"]
                # If this is the case, the network predicts over the whole window at high frequency.
                # TODO utilize the whole window measurements. May improve.
                if meas.dim() == 3:
                    meas = meas[:, -1]
                    meas_cov = meas_cov[:, -1]
                    
            if '_PearsonCov' in arch_type:

                assert meas.dim() == 2
                assert meas_cov.dim() == 3

                meas = meas.cpu().detach()
                meas_cov[meas_cov < -4] = -4
                meas_cov = meas_cov.cpu().detach()

                frame = frame.cpu().detach()

                meas = torch.concat([torch.matmul(frame.permute(0,2,1), meas[:,:2].unsqueeze(-1)).squeeze(-1),meas[:,-1].unsqueeze(-1)], dim=-1)

                frame_3d = torch.cat((torch.cat((frame, torch.zeros(*frame.shape[:-1], 1)), dim=-1), torch.Tensor([0,0,1]).reshape(1,1,3).broadcast_to(frame.shape[0],1,3)), dim=-2)
                meas_cov = torch.matmul(frame_3d.permute(0,2,1),torch.matmul(meas_cov,frame_3d))
                meas_cov = meas_cov.numpy()[0,:,:]
                meas = meas.numpy()
                meas = meas.reshape((3, 1))

            elif 'so3_frame' in arch_type:
                # SO(3) extension: model returns (frame=R^T, disp_canonical, cov_canonical=diag log-std).
                # disp_world  = R @ disp_canonical = frame.T @ disp_canonical
                # Sigma_world = R @ diag(exp(2*log_std)) @ R.T
                assert meas.dim() == 2      # [1, 3] canonical disp
                assert meas_cov.dim() == 2  # [1, 3] canonical diag log-std
                meas = meas.cpu().detach()
                meas_cov = meas_cov.cpu().detach()
                meas_cov[meas_cov < -4] = -4
                frame = frame.cpu().detach()
                meas = torch.einsum('bij,bj->bi', frame.transpose(-2, -1), meas)
                sigma2_diag = torch.exp(2.0 * meas_cov)
                Sigma_can = torch.diag_embed(sigma2_diag)
                Sigma_world = torch.einsum('bij,bjk,blk->bil',
                                            frame.transpose(-2, -1), Sigma_can, frame.transpose(-2, -1))
                meas_cov = Sigma_world.numpy()[0, :, :]
                meas = meas.numpy().reshape((3, 1))

            elif '_fullCov' in arch_type:

                assert meas.dim() == 2
                assert meas_cov.dim() == 2

                meas = meas.cpu().detach()
                meas_cov[meas_cov < -4] = -4
                meas_cov = meas_cov.cpu().detach()
                
                frame = frame.cpu().detach()

                meas = torch.concat([torch.matmul(frame.permute(0,2,1), meas[:,:2].unsqueeze(-1)).squeeze(-1),meas[:,-1].unsqueeze(-1)], dim=-1)

                meas_cov = torch.exp(2*meas_cov)
                new_meas_cov = torch.matmul(frame.permute(0,2,1),torch.matmul(torch.diag_embed(meas_cov[:,:-1], offset=0, dim1=-2, dim2=-1),frame))
                new_meas_cov = torch.cat((new_meas_cov, torch.zeros((*new_meas_cov.shape[:-1],1), device = new_meas_cov.device)),dim=-1) 
                new_meas_cov = torch.cat((new_meas_cov,torch.diag_embed(meas_cov[:,-1].unsqueeze(-1), offset=2, dim1=-2, dim2=-1)[:,0,:].unsqueeze(-2)),dim=-2)   
                meas_cov = new_meas_cov.numpy()[0,:,:]
                meas = meas.numpy()
                meas = meas.reshape((3, 1))

            # elif '_fullCov' in arch_type and '_frameop' not in arch_type:
                
            #     assert meas.dim() == 2
            #     assert meas_cov.dim() == 3

            #     meas = meas.cpu().detach().numpy()
            #     meas_cov[meas_cov < -4] = -4
            #     meas_cov = meas_cov.cpu().detach().numpy()[0, :, :]
            #     meas = meas.reshape((3, 1))

                

            elif 'chol' in arch_type:
                # Cholesky-style covariance head: net returns (mean, A) where Sigma = A A^T + eps*I.
                # See imu-eq-learning ln_resnet_chol. eps matches eq_chol_eps=1e-3.
                assert meas.dim() == 2     # [1,3]
                assert meas_cov.dim() == 3  # [1,3,3]
                A_t = meas_cov.cpu().detach()
                Sigma = torch.matmul(A_t, A_t.transpose(-2, -1)) + 1e-3 * torch.eye(3).unsqueeze(0)
                meas_cov = Sigma.numpy()[0, :, :]
                meas = meas.cpu().detach().numpy().reshape((3, 1))

            else:

                assert meas.dim() == 2  # [B,3]
                assert meas_cov.dim() == 2

                meas = meas.cpu().detach().numpy()
                meas_cov[meas_cov < -4] = -4  # exp(-3) =~ 0.05
                meas_cov = DiagonalParam.vec2Cov(meas_cov).cpu().detach().numpy()[0, :, :]
                meas = meas.reshape((3, 1))

            # Our equivalent of zero position update (TODO need stronger prior to keep it still)
            if clip_small_disp and np.linalg.norm(meas) < 0.001:
                meas = 0 * meas
                # meas_cov = 1e-6 * np.eye(3)

            return meas, meas_cov
