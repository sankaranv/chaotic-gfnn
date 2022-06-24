import os
import sys
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import multiprocessing as mp
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
from orbital import Orb2PosVel  # noqa: E402
from utils import Simulate  # noqa: E402


def QDot(p):
    return p


def PDot(q):
    return -q / (np.linalg.norm(q)**3)


def SimulateTwoBody(pos, vel, step_size, steps):
    return Simulate(pos, vel, step_size, steps, QDot, PDot)


def PartialTrajs(seq_len, step_size, steps_per_h, sigma, n_jobs):
    process = mp.current_process()
    print("Start: processor #", process._identity[0])
    if (process._identity[0] == 1):
        print("Progress on thread 1")
        jobrange = tqdm(range(n_jobs))
    else:
        jobrange = range(n_jobs)
    ret = []
    for _ in jobrange:
        # Randomly initialize orbital elements of Keplerian orbits in the data
        # domain (gravitational constant G is set to be 1).
        a = np.random.uniform(0.8, 1.2)
        e = np.random.uniform(0, 0.05)
        i = np.random.uniform(0, 0)
        Omega = np.random.uniform(0, 2*np.pi)
        omega = np.random.uniform(0, 2*np.pi)
        E = np.random.uniform(0, 2*np.pi)
        orb = [a, e, i, Omega, omega, E]

        # Convert from orbital elements to position and velocity.
        pos, vel = Orb2PosVel(orb, 1.0)
        pos, vel = pos[0:2]+np.random.randn(2)*sigma,\
            vel[0:2]+np.random.randn(2)*sigma
        traj = SimulateTwoBody(pos, vel, step_size, steps_per_h*(seq_len-1))
        ret.append(traj[::steps_per_h, :])
    print("Finish: processor #", process._identity[0])
    return ret


class TwoBodyData(Dataset):
    """ Two body data generator.

    Generate N trajs (length equals seq_len) with initial condition drawn
    from the data domain.

    Attributes:
        seqs: list of {sequence of [posx, posy, velx, vely]}
    """
    def __init__(self, N, h, seq_len=2, n_processors=1, sigma=1e-2, noise=0.0):
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
        n_traj = []
        for pid in range(n_processors):
            n_traj.append(int(N/n_processors))
        n_traj[0] += N-int(N/n_processors)*n_processors
        pool = mp.get_context("spawn").Pool(processes=n_processors)
        results = [pool.apply_async(PartialTrajs,
                                    args=(seq_len, step_size, steps_per_h,
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
        print("==================\n #Generating training data:")
        train_data = TwoBodyData(n_train, h, seq_len=seq_len,
                                 n_processors=n_processors)
        torch.save(train_data, TRAIN_DATA_FILE)
    if os.path.exists(TEST_DATA_FILE):
        print("Testing data already exists.")
    else:
        print("==================\n #Generating testing data:")
        test_data = TwoBodyData(n_test, h, seq_len=seq_len,
                                n_processors=n_processors)
        torch.save(test_data, TEST_DATA_FILE)
    return TRAIN_DATA_FILE, TEST_DATA_FILE
