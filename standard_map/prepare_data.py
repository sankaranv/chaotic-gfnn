import os
import sys
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import multiprocessing as mp
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))


def StandardMap(p, t, K):
    p = p + K * np.sin(t)
    t = t + p
    return p, t


def StandardMapTraj(pt0, K, L):
    p, t = pt0
    ptspan = np.zeros((L+1, 2))
    ptspan[0, :] = pt0
    for i in range(L):
        p, t = StandardMap(p, t, K)
        p = np.fmod(p+2*np.pi, 2*np.pi)
        t = np.fmod(t+2*np.pi, 2*np.pi)
        ptspan[i+1, :] = np.array([p, t])
    return ptspan


def PartialTrajs(base_traj, seq_len, sigma, n_jobs, K):
    # @n_jobs: number of sequences to generate
    process = mp.current_process()
    print("Start: ", process)
    if (process._identity[0] == 1):
        jobrange = tqdm(range(n_jobs))
    else:
        jobrange = range(n_jobs)
    ret = []
    N = base_traj.shape[0]
    for _ in jobrange:
        idx = np.random.randint(N)
        state = base_traj[idx, :] + np.random.randn(2) * sigma
        p, t = np.fmod(state+2*np.pi, 2*np.pi)
        seq = [[p, t]]
        for i in range(seq_len-1):
            p, t = StandardMap(p, t, K)
            seq.append([p, t])
            p = np.fmod(p+2*np.pi, 2*np.pi)
            t = np.fmod(t+2*np.pi, 2*np.pi)
        ret.append(np.array(seq))
    print("Finish: ", process)
    return ret


class StandardMapData(Dataset):
    """ Standard map data generator.

    Generate N trajs (length equals seq_len) with initial condition drawn
    from the data domain.

    Attributes:
        seqs: list of {sequence of [p, t]}
    """
    def __init__(self, N, K, p0, t0, seq_len=2, n_processors=1,
                 sigma=0.5, L=100000, noise=0.0):
        """
        Args:
            N: number of data.
            h: step size of the flow.
            sigma: perturbation scale.
            seq_len: the length of each sequence.
            noise: noise_level of the trajectory.
            n_processor: number of processors used for generating data.
            K: parameter in the standard map
        """

        self.seqs = []
        base_traj = StandardMapTraj(np.array([p0, t0]), K, L)

        n_traj = []
        for pid in range(n_processors):
            n_traj.append(int(N/n_processors))
        n_traj[0] += N-int(N/n_processors)*n_processors
        pool = mp.get_context("spawn").Pool(processes=n_processors)
        results = [pool.apply_async(PartialTrajs,
                                    args=(base_traj, seq_len,
                                          sigma, n_traj[pid], K))
                   for pid in range(n_processors)]
        for e in results:
            for seq in e.get():
                self.seqs.append(seq + np.random.randn(seq_len, 2)*noise)
        pool.close()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.float,
                            requires_grad=True)


def PrepareData(dirname, K, p0, t0, n_train=100000, n_test=1000,
                seq_len=2, n_processors=1, prefix=""):

    if not os.path.exists(dirname):
        os.mkdir(dirname)
    TRAIN_DATA_FILE = dirname + "/" + prefix + "train_K=" + str(K) + "_seq=" \
        + str(seq_len) + ".tensor"
    TEST_DATA_FILE = dirname + "/" + prefix + "test_K=" + str(K) + "_seq=" \
        + str(seq_len) + ".tensor"
    if os.path.exists(TRAIN_DATA_FILE):
        print("Training data already exists!")
    else:
        print("==================\n # Generating training data:")
        train_data = StandardMapData(n_train, K, p0, t0, seq_len, n_processors)
        torch.save(train_data, TRAIN_DATA_FILE)
    if os.path.exists(TEST_DATA_FILE):
        print("Testing data already exists!")
    else:
        print("==================\n # Generating testing data:")
        test_data = StandardMapData(n_test, K, p0, t0, seq_len, n_processors)
        torch.save(test_data, TEST_DATA_FILE)
    return TRAIN_DATA_FILE, TEST_DATA_FILE
