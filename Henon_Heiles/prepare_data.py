import os
import sys
import pathlib
import multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
from utils import Simulate  # noqa: E402


def QDot(p):
    return p


def PDot(q):
    x, y = q[0], q[1]
    return np.array([-x-2*x*y, -y-x*x+y*y])


def SimulateHenonHeiles(pos, vel, step_size, steps):
    return Simulate(pos, vel, step_size, steps, QDot, PDot)


def PartialTrajs(traj, seq_len, step_size, steps_per_h, N, sigma, n_jobs):
    process = mp.current_process()
    print("Start: ", process)
    if (process._identity[0] == 1):
        print("Progress on thread 1")
        jobrange = tqdm(range(n_jobs))
    else:
        jobrange = range(n_jobs)
    ret = []
    for _ in jobrange:
        idx = np.random.randint(N)
        state = traj[idx, :] + np.random.randn(4) * sigma
        q = state[0:2]
        p = state[2:4]
        t = SimulateHenonHeiles(q,
                                p,
                                step_size,
                                steps_per_h*(seq_len-1))
        ret.append(t[::steps_per_h, :])
    print("Finish: ", process)
    return ret


class HenonHeilesData(Dataset):
    """ Henon Heiles data generator.

    Generate N trajs (length equals seq_len) with initial condition drawn
    from the data domain.

    Attributes:
        seqs: list of {sequence of [q1, q2, p1, p2]}
    """
    def __init__(self, q0, p0, N, h, sigma=1e-2, seq_len=2, n_processors=1,
                 T=1000, noise=0.0):
        """
        Args:
            N: number of data.
            h: step size of the flow.
            sigma: perturbation scale.
            seq_len: the length of each sequence.
            noise: noise_level of the trajectory.
            n_processor: number of processors used for generating data.
        """

        self.seqs = []
        steps_per_h = int(h/1e-3)
        step_size = h/steps_per_h
        tot_steps = steps_per_h*int(T/h+1)
        traj = SimulateHenonHeiles(q0,
                                   p0,
                                   step_size,
                                   tot_steps)

        n_traj = []
        for pid in range(n_processors):
            n_traj.append(int(N/n_processors))
        n_traj[0] += N-int(N/n_processors)*n_processors
        pool = mp.get_context("spawn").Pool(processes=n_processors)
        results = [pool.apply_async(PartialTrajs,
                                    args=(traj, seq_len, step_size,
                                          steps_per_h, tot_steps,
                                          sigma, n_traj[pid]))
                   for pid in range(n_processors)]
        for e in results:
            for seq in e.get():
                self.seqs.append(seq)
        pool.close()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.float,
                            requires_grad=True)


def PrepareData(dirname, q0, p0, h=0.1, n_train=100000, n_test=1000,
                seq_len=2, n_processors=1, prefix=""):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    TRAIN_DATA_FILE = dirname + "/" + prefix + "train_h=" + str(h) +\
        "_seq_len=" + str(seq_len) + ".tensor"
    TEST_DATA_FILE = dirname + "/" + prefix + "test_h=" + str(h) +\
        "_seq_len=" + str(seq_len) + ".tensor"
    if os.path.exists(TRAIN_DATA_FILE):
        print("Training data already exists.")
    else:
        print("==================\n # Generating training data:")
        train_data = HenonHeilesData(q0, p0, n_train, h, seq_len=seq_len,
                                     n_processors=n_processors)
        torch.save(train_data, TRAIN_DATA_FILE)
    if os.path.exists(TEST_DATA_FILE):
        print("Testing data already exists.")
    else:
        print("==================\n # Generating testing data:")
        test_data = HenonHeilesData(q0, p0, n_test, h,
                                    seq_len=seq_len, n_processors=n_processors)
        torch.save(test_data, TEST_DATA_FILE)
    return TRAIN_DATA_FILE, TEST_DATA_FILE
