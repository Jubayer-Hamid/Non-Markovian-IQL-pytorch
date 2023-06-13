import torch
import torch.nn as nn
import torch.distributions
from common import MLP
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions import MultivariateNormal

class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
        self, state_dim, action_dim, hidden_dim, n_layers, dropout_rate=None,
        log_std_min=-10.0, log_std_max=2.0,
    ):
        super().__init__()

        # self.mlp = MLP(
        #     state_dim, 2 * action_dim, hidden_dim, n_layers, dropout_rate=dropout_rate
        # )

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, dropout=0.2)
        self.lstm_dense = nn.Linear(hidden_dim, action_dim)
        self.state_dependent_std_dense = nn.Linear(hidden_dim, action_dim)

        self.log_stds = nn.Parameter(torch.zeros(action_dim,))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(
        self, states
    ):
        # mu, log_std = self.mlp(states).chunk(2, dim=-1)

        # mu = torch.tanh(mu) # good
        outputs = self.lstm(states)[0]
        # print(f'outputs.shape {outputs.shape}')
        means = self.lstm_dense(outputs)
        # print(f'means.shape {means.shape}')
        log_stds = self.log_stds #self.state_dependent_std_dense(outputs)
        
        log_stds = torch.clip(log_stds, self.log_std_min, self.log_std_max)

        means = torch.tanh(means)

        scale_diag = torch.exp(log_stds)

        # print(f'means: {means.shape}; scale_diag: {scale_diag.shape}')
        base_dist = Independent(Normal(means, scale_diag), 1)
        
        # base_dist = tfd.MultivariateNormalDiag(loc=means,
        #                                        scale_diag=jnp.exp(log_stds) *
        #                                                   temperature)        

        # print(f'base_dist: {base_dist}')

        return base_dist

    def get_action(self, states):
        dist = self.forward(states)
        action = dist.sample()
        return action


class Actor_non_independent_MVN(nn.Module):
    """MLP actor network."""

    def __init__(
        self, state_dim, action_dim, hidden_dim, n_layers, dropout_rate=None,
        log_std_min=-10.0, log_std_max=2.0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, dropout=0.2)
        self.lstm_dense = nn.Linear(hidden_dim, action_dim)
        self.state_dependent_std_dense = nn.Linear(hidden_dim, action_dim)

        self.mean_linear = nn.Linear(action_dim, action_dim)
        self.log_std_linear = nn.Linear(action_dim, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, states):
        outputs = self.lstm(states)[0]
        means = self.lstm_dense(outputs)
        log_stds = self.state_dependent_std_dense(outputs)

        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
        
        means = torch.tanh(means)
        log_stds = torch.tanh(log_stds)

        cov_matrix = torch.diag_embed(torch.exp(log_stds))
        action_distribution = MultivariateNormal(means, cov_matrix)

        return action_distribution

    def get_action(self, states):
        dist = self.forward(states)
        action = dist.sample()
        return action

class Actor_transformer(nn.Module):
    """MLP actor network."""

    def __init__(
        self, state_dim, action_dim, hidden_dim, n_layers, dropout_rate=None,
        log_std_min=-10.0, log_std_max=2.0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, dropout=0.2)
        self.lstm_dense = nn.Linear(hidden_dim, action_dim)
        self.state_dependent_std_dense = nn.Linear(hidden_dim, action_dim)

        self.mean_linear = nn.Linear(action_dim, action_dim)
        self.log_std_linear = nn.Linear(action_dim, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, states):
        outputs = self.lstm(states)[0]
        means = self.lstm_dense(outputs)
        log_stds = self.state_dependent_std_dense(outputs)

        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
        
        means = torch.tanh(means)
        log_stds = torch.tanh(log_stds)

        cov_matrix = torch.diag_embed(torch.exp(log_stds))
        action_distribution = MultivariateNormal(means, cov_matrix)

        return action_distribution

    def get_action(self, states):
        dist = self.forward(states)
        action = dist.sample()
        return action


class Actor_transformer(nn.Module):
    """Transformer-based actor network."""

    def __init__(
        self, state_dim, action_dim, hidden_dim, n_layers=3, dropout_rate=None,
        log_std_min=-10.0, log_std_max=2.0,
    ):
        super().__init__()

        self.transformer = nn.Transformer(
            d_model=state_dim,
            nhead=4,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
        )

        self.mlp = MLP(
            state_dim, 2 * action_dim, hidden_dim, n_layers, dropout_rate=dropout_rate
        )

        self.log_stds = nn.Parameter(torch.zeros(action_dim,))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, states):
        # Apply transformer encoding
        encoded_states = self.transformer(states)

        # Apply MLP to encoded states
        mu, log_std = self.mlp(encoded_states).chunk(2, dim=-1)

        mu = torch.tanh(mu)

        # Clamp log standard deviations
        log_std = torch.clip(log_std, self.log_std_min, self.log_std_max)

        scale_diag = torch.exp(log_std)

        base_dist = Independent(Normal(mu, scale_diag), 1)

        return base_dist

    def get_action(self, states):
        dist = self.forward(states)
        action = dist.sample()
        return action
