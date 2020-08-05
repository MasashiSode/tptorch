from gpytorch.mlls import MarginalLogLikelihood

from ..distributions import MultivariateStudentT
from ..likelihoods import _StudentTLikelihoodBase


class ExactStudentTMarginalLogLikelihood(MarginalLogLikelihood):
    def __init__(self, likelihood, model):
        if not isinstance(likelihood, _StudentTLikelihoodBase):
            raise RuntimeError("Likelihood must be student t for exact inference")
        super(ExactStudentTMarginalLogLikelihood, self).__init__(likelihood, model)

    def forward(self, function_dist, target, *params):
        if not isinstance(function_dist, MultivariateStudentT):
            raise RuntimeError(
                "StudentTExactMarginalLogLikelihood can only operate on student t random variables"
            )

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)
        res = output.log_prob(target)

        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        for _, prior, closure, _ in self.named_priors():
            res.add_(prior.log_prob(closure()).sum())

        # Scale by the amount of data we have
        num_data = target.size(-1)
        return res.div_(num_data)

    def pyro_factor(self, output, target, *params):
        import pyro

        mll = self(output, target, *params)
        pyro.factor("gp_mll", mll)
        return mll
