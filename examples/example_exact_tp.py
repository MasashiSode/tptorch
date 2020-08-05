import math
import os
import random

import gpytorch
import numpy as np
import pyro
import torch
from gpytorch import kernels
from matplotlib import pyplot as plt

from tptorch.distributions import MultivariateStudentT
from tptorch.likelihoods import StudentTLikelihood
from tptorch.mlls import ExactStudentTMarginalLogLikelihood
from tptorch.models import ExactTP

SEED = 2020


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    pyro.set_rng_seed(SEED)


seed_everything(SEED)


class TPModel(ExactTP):
    def __init__(
        self, train_x, train_y, likelihood, nu=5,
    ):
        super(TPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        nu = torch.tensor(nu, dtype=torch.float64)
        self.nu = torch.nn.Parameter(nu)
        self.data_num = torch.tensor(self.train_targets.shape[0], dtype=torch.float64)

        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1)

        self.covar_module = kernels.ScaleKernel(base_kernel=base_kernel, num_dims=1,)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_x_train_data = self.covar_module(self.train_inputs[0])
        inv_quad, _ = covar_x_train_data.inv_quad_logdet(
            inv_quad_rhs=self.train_targets - self.train_targets.mean(), logdet=False
        )

        tp_var_scale = (self.nu + inv_quad - 2) / (self.nu + self.data_num - 2)

        covar_x = tp_var_scale.float() * covar_x

        return MultivariateStudentT(mean_x, covar_x, self.nu, self.data_num)


if __name__ == "__main__":
    train_x = (torch.linspace(0, 1, 10)) ** 1
    train_y = (
        torch.sin(train_x * (2 * math.pi))
        + torch.randn(train_x.size()) * math.sqrt(0.1)
        + (torch.randn(train_x.size()) >= (1 - 0.1)) * torch.randn(train_x.size()) * 3
    )

    nu = 5
    likelihood = StudentTLikelihood()
    model = TPModel(train_x, train_y, likelihood, nu=nu)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()},], lr=0.1  # Includes GaussianLikelihood parameters
    )
    # "Loss" for GPs - the marginal log likelihood
    mll = ExactStudentTMarginalLogLikelihood(likelihood, model)

    training_iter = 100
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print(
            "Iter %d/%d - Loss: %.3f   lengthscale: %.3f"
            % (
                i + 1,
                training_iter,
                loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                # model.likelihood.noise.item(),
            )
        )
        optimizer.step()
        with torch.no_grad():
            model.nu.clamp_(2)
    print(model.nu)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cg_iterations(
        2000
    ):
        valid_x = torch.linspace(0, 1, 51)
        valid_y = torch.sin(valid_x * (2 * math.pi))

        output_valid = model(valid_x)
        observed_pred_valid = likelihood(output_valid)

        mean_valid = observed_pred_valid.mean
        mean_valid = mean_valid.cpu()
    with torch.no_grad():

        gt_x = torch.linspace(0, 1, 100)
        gt_y = torch.sin(gt_x * (2 * math.pi))
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred_valid.confidence_region()
        # Plot training data as black stars
        ax.plot(gt_x.numpy(), gt_y.numpy(), "r")
        ax.plot(train_x.numpy(), train_y.numpy(), "k*")
        # Plot predictive means as blue line
        ax.plot(valid_x.numpy(), observed_pred_valid.mean.numpy(), "b")
        # Shade between the lower and upper confidence bounds
        ax.fill_between(valid_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-4, 4])
        ax.legend(["gt", "Observed Data", "tp Mean", "tp Confidence"])
        plt.savefig("tproces_test.png")
