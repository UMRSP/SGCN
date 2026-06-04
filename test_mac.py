import os
import pickle
import glob
import copy
from time import time

import torch
import torch.distributions.multivariate_normal as torchdist
from torch.utils.data.dataloader import DataLoader

from utils import *
from metrics import *
from model_mac import TrajectoryModel


# ============================================================
# Device selection (MPS -> CUDA -> CPU)
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


def test(model, loader_test, KSTEPS=20):

    model.eval()

    raw_data_dict = {}
    ade_bigls = []
    fde_bigls = []

    avg_time = []
    avg_batch = []

    step = 0

    with torch.no_grad():

        for batch in loader_test:

            step += 1

            batch = [tensor.to(DEVICE) for tensor in batch]

            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                V_obs,
                V_tr
            ) = batch

            avg_batch.append(obs_traj.shape[1])

            identity_spatial = (
                torch.ones(
                    (V_obs.shape[1], V_obs.shape[2], V_obs.shape[2]),
                    device=DEVICE
                )
                * torch.eye(V_obs.shape[2], device=DEVICE)
            )

            identity_temporal = (
                torch.ones(
                    (V_obs.shape[2], V_obs.shape[1], V_obs.shape[1]),
                    device=DEVICE
                )
                * torch.eye(V_obs.shape[1], device=DEVICE)
            )

            identity = [identity_spatial, identity_temporal]

            start_time_inference = time()

            V_pred = model(V_obs, identity)

            avg_time.append(time() - start_time_inference)

            V_pred = V_pred.squeeze()
            V_tr = V_tr.squeeze()

            num_of_objs = obs_traj_rel.shape[1]

            V_pred = V_pred[:, :num_of_objs, :]
            V_tr = V_tr[:, :num_of_objs, :]

            sx = torch.exp(V_pred[:, :, 2])
            sy = torch.exp(V_pred[:, :, 3])
            corr = torch.tanh(V_pred[:, :, 4])

            cov = torch.zeros(
                V_pred.shape[0],
                V_pred.shape[1],
                2,
                2,
                device=DEVICE
            )

            cov[:, :, 0, 0] = sx * sx
            cov[:, :, 0, 1] = corr * sx * sy
            cov[:, :, 1, 0] = corr * sx * sy
            cov[:, :, 1, 1] = sy * sy

            mean = V_pred[:, :, 0:2]

            mvnormal = torchdist.MultivariateNormal(mean, cov)

            V_x = seq_to_nodes(obs_traj.cpu().numpy().copy())

            V_x_rel_to_abs = nodes_rel_to_nodes_abs(
                V_obs[:, :, :, :2].cpu().numpy().squeeze().copy(),
                V_x[0, :, :].copy()
            )

            V_y = seq_to_nodes(pred_traj_gt.cpu().numpy().copy())

            V_y_rel_to_abs = nodes_rel_to_nodes_abs(
                V_tr.cpu().numpy().squeeze().copy(),
                V_x[-1, :, :].copy()
            )

            raw_data_dict[step] = {}
            raw_data_dict[step]["obs"] = copy.deepcopy(V_x_rel_to_abs)
            raw_data_dict[step]["trgt"] = copy.deepcopy(V_y_rel_to_abs)
            raw_data_dict[step]["pred"] = []

            ade_ls = {}
            fde_ls = {}

            for n in range(num_of_objs):
                ade_ls[n] = []
                fde_ls[n] = []

            for k in range(KSTEPS):

                V_pred_sample = mvnormal.sample()

                V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                    V_pred_sample.cpu().numpy().squeeze().copy(),
                    V_x[-1, :, :].copy()
                )

                raw_data_dict[step]["pred"].append(
                    copy.deepcopy(V_pred_rel_to_abs)
                )

                for n in range(num_of_objs):

                    pred = []
                    target = []
                    obsrvs = []
                    number_of = []

                    pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                    target.append(V_y_rel_to_abs[:, n:n + 1, :])
                    obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                    number_of.append(1)

                    ade_ls[n].append(ade(pred, target, number_of))
                    fde_ls[n].append(fde(pred, target, number_of))

            for n in range(num_of_objs):
                ade_bigls.append(min(ade_ls[n]))
                fde_bigls.append(min(fde_ls[n]))

    print("AVG trajectories per batch ", np.mean(avg_batch))
    print("AVG inference time ", np.mean(avg_time))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)

    return ade_, fde_, raw_data_dict


def main():

    KSTEPS = 20

    ade_ls = []
    fde_ls = []

    print("Number of samples:", KSTEPS)
    print("*" * 50)

    root_ = "./checkpoints/"

    dataset = ["eth", "hotel", "univ", "zara1", "zara2"]

    paths = list(map(lambda x: root_ + x, dataset))

    for feta in range(len(paths)):

        path = paths[feta]

        exps = glob.glob(path)

        print("Model being tested are:", exps)

        for exp_path in exps:

            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = exp_path + "/val_best.pth"
            args_path = exp_path + "/args.pkl"

            with open(args_path, "rb") as f:
                args = pickle.load(f)

            obs_seq_len = args.obs_len
            pred_seq_len = args.pred_len

            data_set = "./dataset/" + args.dataset + "/"

            dset_test = TrajectoryDataset(
                data_set + "test/",
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1
            )

            start_time_loader = time()

            loader_test = DataLoader(
                dset_test,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )

            print("Finished loader time ", time() - start_time_loader)

            start_time_model = time()

            model = TrajectoryModel(
                number_asymmetric_conv_layer=7,
                embedding_dims=64,
                number_gcn_layers=1,
                dropout=0,
                obs_len=8,
                pred_len=12,
                n_tcn=5,
                out_dims=5
            ).to(DEVICE)

            model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=DEVICE
                )
            )

            model.eval()

            print("Finished model time ", time() - start_time_model)

            print("Testing ....")

            ade_, fde_, raw_data_dict = test(
                model,
                loader_test,
                KSTEPS=KSTEPS
            )

            ade_ls.append(ade_)
            fde_ls.append(fde_)

            print("ade:", ade_, " fde:", fde_)

        print("*" * 50)

    print("Avg ADE:", sum(ade_ls) / 5)
    print("Avg FDE:", sum(fde_ls) / 5)


if __name__ == "__main__":
    main()