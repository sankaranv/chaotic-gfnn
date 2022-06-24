import os
import sys
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import multiprocessing as mp
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
from utils import Simulate  # noqa: E402


def QDot(q, p):
    x, y = q
    px, py = p
    return np.array([px+y, py-x])


def PDot(q, p):
    mu = 0.5
    x, y = q
    px, py = p
    r1, r2 = np.sqrt((x+mu)*(x+mu)+y*y), np.sqrt((x+mu-1)*(x+mu-1)+y*y)
    return -np.array([-py+(1-mu)*(x+mu)/r1/r1/r1+mu*(x+mu-1)/r2/r2/r2,
                      px+(1-mu)*y/r1/r1/r1+mu*y/r2/r2/r2])


def SimulatePCR3BP(pos, vel, step_size, steps):
    return Simulate(pos, vel, step_size, steps, QDot, PDot, method="RK4")


def PartialTrajs(traj, seq_len, step_size, steps_per_h, N, sigma, n_jobs):
    process = mp.current_process()
    print("Start: processor #", process._identity[0])
    if (process._identity[0] == 1):
        print("Progress on thread 1")
        jobrange = tqdm(range(n_jobs))
    else:
        jobrange = range(n_jobs)
    ret = []
    for _ in jobrange:
        idx = np.random.randint(N)
        state = traj[idx, :]
        # Perturb IC: q is more sensitive than p
        q = state[0:2] + np.random.randn(2) * sigma
        p = state[2:4] + np.random.randn(2) * sigma
        t = SimulatePCR3BP(q, p, step_size, steps_per_h*(seq_len-1))
        ret.append(t[::steps_per_h, :])
    print("Finish: processor #", process._identity[0])
    return ret


class PCR3BPData(Dataset):
    """ PCR3BP data generator.

    Generate N trajs (length equals seq_len) with initial condition drawn
    from the data domain.

    Attributes:
        seqs: list of {sequence of [q1, q2, p1, p2]}
    """
    def __init__(self, N, h, seq_len=2, n_processors=1,
                 sigma=5e-2, T=1000, noise=0.0):
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

        q0, p0 = np.array([5, 0]), np.array([0, 1/np.sqrt(4.5)])
        steps_per_h = int(h/1e-3)
        step_size = h/steps_per_h
        tot_steps = steps_per_h*int(T/h+1)
        traj = SimulatePCR3BP(q0, p0, step_size, tot_steps)

        step_size = h/steps_per_h
        n_processor = 10
        n_traj = []
        for pid in range(n_processor):
            n_traj.append(int(N/n_processor))
        n_traj[0] += N-int(N/n_processor)*n_processor
        pool = mp.get_context("spawn").Pool(processes=n_processor)
        results = [pool.apply_async(PartialTrajs,
                                    args=(traj, seq_len, step_size,
                                          steps_per_h, tot_steps,
                                          sigma, n_traj[pid]))
                   for pid in range(n_processor)]
        for e in results:
            for seq in e.get():
                self.seqs.append(seq)
        pool.close()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.float,
                            requires_grad=True)


def PrepareData(dirname, h=0.1, n_train=100000, n_test=1000, seq_len=2,
                n_processors=1):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    prefix = "/"
    TRAIN_DATA_FILE = dirname + prefix + "train_h=" + str(h) +\
        "_seq_len=" + str(seq_len) + ".tensor"
    TEST_DATA_FILE = dirname + prefix + "test_h=" + str(h) +\
        "_seq_len=" + str(seq_len) + ".tensor"
    if os.path.exists(TRAIN_DATA_FILE):
        print("Training data already exists.")
    else:
        print("==================\n# Generating training data:")
        train_data = PCR3BPData(n_train, h, seq_len=seq_len,
                                n_processors=n_processors)
        torch.save(train_data, TRAIN_DATA_FILE)
    if os.path.exists(TEST_DATA_FILE):
        print("Testing data already exists.")
    else:
        print("==================\n# Generating testing data:")
        test_data = PCR3BPData(n_test, h, seq_len=seq_len,
                               n_processors=n_processors)
        torch.save(test_data, TEST_DATA_FILE)
    return TRAIN_DATA_FILE, TEST_DATA_FILE
