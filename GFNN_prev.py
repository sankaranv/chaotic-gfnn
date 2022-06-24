from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from scipy.optimize import newton

class Model(nn.Module):
    def __init__(self, dims):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1], bias=True)
        self.fc2 = nn.Linear(dims[1], dims[2], bias=True)
        self.fc3 = nn.Linear(dims[2], dims[3], bias=True)
        self.fc4 = nn.Linear(dims[3], dims[4], bias=True)
        self.fc5 = nn.Linear(dims[4], dims[5], bias=True)

        for fc in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            torch.nn.init.orthogonal_(fc.weight)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x


class GFNN:
    """Summary of class GFNN (generating function neural network).

    GFNN is specialized in predicting symplectic data sequences using neural
    networks.

    Attributes:
        args: a dict of arguments:
            {
                b"n_neurons": (dimensions of all 5 layers of the model),
                b"dim": (the input dimension of p (q),
                    hence the dimension of the first layer of model is dim*2),
                b"train_file_path": (path of the training data),
                b"test_file_path": (path of the testing data),
                b"batch_size": (batch size),
                b"step_size": (step size of one step evolution)
            }
        model: the feedforwork neural network.
        epoch: current epoch.
        loss_array: training loss.
        test_loss_array: testing loss.
    """
    def __init__(self, args={}):
        """Init GFNN."""

        self.ParseArgs(args)
        self.model = Model(self.args["n_neurons"])
        self.LoadData()
        self.InitOptimizer()
        self.criterion = nn.MSELoss()
        self.epoch = 0
        self.loss_array = []
        self.test_loss_array = []

        # assign device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

    def ParseArgs(self, args):
        """Parse input args."""

        # Check required arguments.
        assert "step_size" in args, "Step size (h) must be provided."
        assert "dim" in args, "Dimension of p / q must be provided."
        assert "train_file_path" in args,\
            "Path of training data must be provided."
        assert "test_file_path" in args,\
            "Path of testing data must be provided."
        assert "lr" in args, "Learning rate (lr) must be provided."

        # Set to default if the corresponding arguments are not given.
        if "n_neurons" not in args:
            args["n_neurons"] = [args["dim"]*2, 200, 200, 100, 50, 1]
        assert args["n_neurons"][0] == args["dim"]*2,\
            "Incorrect dimension of the first layer."
        # Parse batch size, if not given, use batch_size=100.
        if "batch_size" not in args:
            args["batch_size"] = 100

        self.args = args

    def LoadData(self):
        """Load data."""

        # load traning, testing data
        train_data = torch.load(self.args["train_file_path"])
        test_data = torch.load(self.args["test_file_path"])
        self.n_train, self.n_test = len(train_data), len(test_data)
        batch_size = int(self.args["batch_size"])

        # prepare data_loaders
        self.train_loader = DataLoader(train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(test_data,
                                      batch_size=self.n_test,
                                      shuffle=False)

    def InitOptimizer(self):
        # initialize loss function and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), self.args["lr"])
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.95)

    def FetchPairs(self, qp0, qp1):
        """Fetch qP, Pq, pQ from qp0(q, p) -> qp1(Q, P)"""
        dim = self.args["dim"]
        device = self.device
        n = qp0.shape[0]
        qP = torch.cat(
            (qp0[:, 0:dim].reshape(
                n, dim), qp1[:, dim:2 * dim].reshape(n, dim)),
            1).clone().detach().requires_grad_(True).to(device)
        Pq = torch.cat(
            (qp1[:, dim:2 * dim].reshape(
                n, dim), qp0[:, 0:dim].reshape(n, dim)),
            1).clone().detach().requires_grad_(True).to(device)
        pQ = torch.cat(
            (qp0[:, dim:2 * dim].clone().detach().reshape(n, dim),
                qp1[:, 0:dim].clone().detach().reshape(n, dim)),
            1).to(device)
        return qP, Pq, pQ

    def Train(self, epochs):

        h = self.args["step_size"]

        t = trange(epochs, leave=True)
        for i in t:
            t.refresh()
            self.epoch = self.epoch + 1
            for batch_idx, (seq) in enumerate(self.train_loader):
                for j in range(seq.shape[1]-1):
                    # @qp0: q, p
                    # @qp1: Q, P
                    # one step evolution (q, p) -> (Q, P)
                    qp0 = torch.squeeze(seq[:, j, :])
                    qp1 = torch.squeeze(seq[:, j+1, :])
                    qP, Pq, pQ = self.FetchPairs(qp0, qp1)
                    grad_F, = torch.autograd.grad(self.model(qP).sum(),
                                                  qP,
                                                  create_graph=True)
                    self.optimizer.zero_grad()
                    loss = self.criterion(pQ,
                                          torch.add(torch.mul(grad_F, h), Pq))
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    self.loss_array.append(loss.item())
            t.set_description(f"Epoch {self.epoch}: " +
                              "training loss={:.2e}, "
                              .format(self.loss_array[-1]) +
                              "testing loss={:.2e}"
                              .format(self.Test()))

        return self.loss_array[-1]

    def Test(self):

        h = self.args["step_size"]

        for batch_idx, (seq) in enumerate(self.test_loader):
            for j in range(seq.shape[1]-1):
                # @qp0: q, p
                # @qp1: Q, P
                # one step evolution (q, p) -> (Q, P)
                qp0 = torch.squeeze(seq[:, j, :])
                qp1 = torch.squeeze(seq[:, j+1, :])
                qP, Pq, pQ = self.FetchPairs(qp0, qp1)
                grad_F, = torch.autograd.grad(self.model(qP).sum(),
                                              qP,
                                              create_graph=True)
                loss = self.criterion(pQ, torch.add(torch.mul(grad_F, h), Pq))
                self.test_loss_array.append(loss.item())
        return self.test_loss_array[-1]

    def Predict(self, q, p, tol=1e-6):
        h = self.args["step_size"]
        dim = self.args["dim"]

        def diff_q(P):
            qP = torch.tensor([q, P], dtype=torch.float,
                              requires_grad=True).reshape(1, dim * 2).\
                to(next(self.model.parameters()).device)
            grad_F, = torch.autograd.grad(self.model(qP).sum(), qP)
            return np.array(
                P) + h * grad_F[0, 0:dim].cpu().detach().numpy() - np.array(p)

        [P_, converged, *zero_der] = newton(diff_q,
                                            x0=p,
                                            tol=tol,
                                            full_output=True,
                                            disp=False)

        if (np.isscalar(converged)):
            if not converged:
                raise RuntimeError("Not converge.")
            else:
                if not converged[0]:
                    raise RuntimeError("Not converge.")
        qP = torch.tensor([q, P_], dtype=torch.float,
                          requires_grad=True).reshape(1, dim * 2).\
            to(next(self.model.parameters()).device)
        grad_F, = torch.autograd.grad(self.model(qP).sum(), qP)
        Q_ = q + h * grad_F[0, dim:2 * dim].cpu().detach().numpy()
        return Q_, P_

    def Save(self, dir_name, file_name="checkpoint.pt"):
        """Save GFNN."""
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_array": self.loss_array,
                "args": self.args
            }, dir_name + "/" + file_name)

    def Load(self, dir_name, file_name="checkpoint.pt"):
        """Load GFNN."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        checkpoint = torch.load(dir_name + "/" + file_name)
        self.args = checkpoint["args"]
        self.epoch = checkpoint["epoch"]
        self.loss_array = checkpoint["loss_array"]
        self.model = Model(self.args["n_neurons"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.LoadData()
        self.criterion = nn.MSELoss()
        self.InitOptimizer()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
