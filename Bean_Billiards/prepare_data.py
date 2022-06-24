import os
import sys
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from matplotlib import pyplot as plt
import multiprocessing as mp
from boundary import *
import rootfinding as rf

np.set_printoptions(precision=7, suppress=True)
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

def billiard_propagator_gen(r, v, bdy):
    s = bdy.linear_intersect_cart(r, v)
    if np.any(np.sign(bdy.coords_cart(s) - r) != np.sign(v)):
        # Then we found the intercept in the wrong direction! Try again.
        try:
            # the intersect_param method excludes passed root s if
            s = bdy.linear_intersect_param(s, v)
        except RuntimeError as e:
            print (e)
    while True:
        tangent = bdy.tangent_cart(s)
        v_tan = np.dot(v, tangent)
        v = 2 * v_tan * tangent - v
        yield s, v_tan, v
        try:
            s = bdy.linear_intersect_param(s, v)
        except RuntimeError as e:
            print (e)

def init_state(r, v, boundary):
    """Initialize state given i.c.'s, position r and velocity v, for
        the given boundary.
    """
    state = {'s_list': [],
             'v_tan_list': [],
             'velocity': np.c_[v],
             'trajectory': np.c_[r],
             'propagator': billiard_propagator_gen(r, v, boundary),
             'boundary': boundary}
    return state


def propagate(state, bounces):
    """Update state by a number 'bounces' of collisions.
    """
    new_s, new_v_tan, new_v = zip(*(next(state['propagator']) #zip(*(state['propagator'].next()
                             for _ in range(int(bounces))))
    state['s_list'].extend(new_s)
    state['v_tan_list'].extend(new_v_tan)
    state['trajectory'] = np.c_[state['trajectory'],
                                state['boundary']\
                                .coords_cart(np.array(new_s))]
    new_v = np.array(new_v).T
    state['velocity'] = np.c_[state['velocity'], new_v]

def propagate_plots(fig, state, n, traj_line=None, poinc_ax=None,
                    clear_poinc_ax=True, **plot_kwargs):
    """Update plots to show 'n' collisions of given state.
    
    Optional: traj_line, the plotline of the trajectory to update;
              poinc_ax, the axis of the Poincare section to update.
    If the state does not know n collisions, first propagate it
     to the necessary number.
    """
    clr = clear_poinc_ax and poinc_ax is not None
    if clr: 
        for item in poinc_ax.collections: item.remove()
        del poinc_ax.collections[:]
    missing = n+1 - state['trajectory'].shape[1]
    if missing <= 0:
        if traj_line is not None:
            traj_line.set_xdata(state['trajectory'][0, 0:n+1])
            traj_line.set_ydata(state['trajectory'][1, 0:n+1])
        if poinc_ax is not None:
            poinc_ax.scatter(state['s_list'][0:n+1],
                             state['v_tan_list'][0:n+1],
                             **plot_kwargs)
    else:
        propagate(state, missing)   
        if traj_line is not None: 
            traj_line.set_xdata(state['trajectory'][0])
            traj_line.set_ydata(state['trajectory'][1])
        if poinc_ax is not None:
            poinc_ax.scatter(state['s_list'],
                             state['v_tan_list'],
                             **plot_kwargs)
    if clr:
        poinc_ax.set_ylim([-1, 1])
        poinc_ax.set_xlim([0, 2*np.pi])
    fig.canvas.draw()


def reset_plots(fig, r, v, boundary, n, traj_line=None,
                poinc_ax=None, **propagate_plots_kwargs):
    """Reinitialize data for given initial conditions.
    
    Optional: traj_line, the plotline of the trajectory to update;
              poinc_ax, the axis of the Poincare section to update.
    """
    state = init_state(r, v, boundary)
    propagate_plots(fig, state, n, traj_line, poinc_ax,
                    **propagate_plots_kwargs)
    return state

def compare_traj_for_bdys(bdys, bounces, r0, theta):
    """Create side-by-side plots of trajectories with the same i.c.'s
        but different boundarys."""
    fig, axlist = plt.subplots(1, len(bdys),
                               figsize=np.array([16,6])*figscale)
    fig.suptitle(r'Trajectories for $r_0$={0}, $\theta_0$={1}'
                 .format(r0, theta))
    v0 = np.array([np.cos(theta), np.sin(theta)])
    for bdy, ax in zip(bdys, axlist):
        bdyline = bdy.coords_cart(np.arange(0, 2*np.pi, 0.01))
        ax.plot(bdyline[0], bdyline[1])
        traj_line, = ax.plot([0, 0], [0, 0], 'o-')
        # Initialize plotline data
        state = reset_plots(fig, r0, v0, bdy, bounces, traj_line)
        ax.set_title(str(bdy))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_aspect('equal')
        ax.set_ylim([-1, 1])
    plt.show()



class BilliardData(Dataset):
    """ Bean Billiard data generator.

    Generate N trajs (length equals seq_len) with initial condition drawn
    from the data domain.

    Attributes:
        seqs: list of {sequence of [p, t]}
    """
    def __init__(self, bdy, N, seq_len=2):
        """
        Args:
            N: number of data.
            bdy: boundary object (BeanBoundary or UnitCircleBoundary)
            seq_len: the length of each sequence.
            r0: initial condition for billiard process
            theta: initial condition for billiard process
        """
        self.seqs = []
        for n in tqdm(range(N)):

            # Set initial conditions - position r and angle theta
            theta = np.random.uniform(low=0, high=np.pi)
            r0 = [np.random.uniform(-0.5,0.5), np.random.uniform(-0.5,0.5)]
            # Generate seq_len steps of the billiard trajectory
            v0 = np.array([np.cos(theta), np.sin(theta)])
            self.bdyline = bdy.coords_cart(np.arange(0, 2*np.pi, 0.01))
            state = init_state(r0, v0, bdy)
            missing = seq_len - state['trajectory'].shape[1]
            if missing > 0:
                propagate(state, missing)
            pos_trajectory = state['trajectory'].T
            v_trajectory = state['velocity'].T
            self.seqs.append(np.c_[pos_trajectory, v_trajectory])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.float,
                            requires_grad=True)

def PrepareData(dirname, a,b,c,d, n_train=100000, n_test=1000,
                seq_len=2, prefix="", force_data_generation = True, **custom_rf):

    bdy = BeanBoundary(a,b,c,d, **custom_rf)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    param_setting_name = f"({a},{b},{c},{d})"
    TRAIN_DATA_FILE = dirname + "/" + prefix + "train_" + param_setting_name + "_seq=" \
        + str(seq_len) + ".tensor"
    TEST_DATA_FILE = dirname + "/" + prefix + "test_" + param_setting_name + "_seq=" \
        + str(seq_len) + ".tensor"
    if os.path.exists(TRAIN_DATA_FILE) or not force_data_generation:
        print("Training data already exists!")
    else:
        print("==================\n # Generating training data:")
        train_data = BilliardData(bdy = bdy, N=n_train, seq_len = seq_len)
        torch.save(train_data, TRAIN_DATA_FILE)
    if os.path.exists(TEST_DATA_FILE) or not force_data_generation:
        print("Testing data already exists!")
    else:
        print("==================\n # Generating testing data:")
        test_data = BilliardData(bdy = bdy, N=n_test, seq_len = seq_len)
        torch.save(test_data, TEST_DATA_FILE)
    return TRAIN_DATA_FILE, TEST_DATA_FILE

if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True)
    figscale = 0.75
    bbdy = BeanBoundary(0.16, 0.1, 2.0, 2.0, **custom_rf)
    data = BilliardData(bdy = bbdy, N=50, seq_len = 10, r0=np.array([0.6, 0.2]), theta=0.1)
    print(data[0].shape)