import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as, to_torch
from torch import nn
from torch.distributions import kl_divergence

# Assuming BasePolicy and DummyLogger are importable from fsrl
# If not, adjust the import path accordingly
try:
    from fsrl.policy import BasePolicy
    from fsrl.utils import BaseLogger, DummyLogger
except ImportError:
    print("Please ensure fsrl library is installed and import paths are correct.")
    # Define dummy classes if fsrl is not available, for the code to be syntactically valid
    class BasePolicy(nn.Module):
        def __init__(self, actor=None, critics=None, *args, **kwargs):
             super().__init__()
             # Simulate BasePolicy setting these if passed
             self.actor = actor
             self.critics = critics if isinstance(critics, nn.ModuleList) else nn.ModuleList([critics]) if critics is not None else nn.ModuleList()
             self.critics_num = len(self.critics)
             self.logger = kwargs.get('logger', DummyLogger()) # Ensure logger is set
        def compute_nstep_returns(self, *args, **kwargs): pass
        def map_action(self, act): return act # Placeholder
        def map_action_inverse(self, act): return act # Placeholder
        def soft_update(self, *args, **kwargs): pass

    class BaseLogger:
        def info(self, *args, **kwargs): print(*args, **kwargs)
        def debug(self, *args, **kwargs): print(*args, **kwargs)
        def warning(self, *args, **kwargs): print("WARNING:", *args, **kwargs)
        def error(self, *args, **kwargs): print("ERROR:", *args, **kwargs)
        def store(self, *args, **kwargs): pass
        def get_latest_scalars(self, *args, **kwargs): return {}

    class DummyLogger(BaseLogger): pass


class CVPO(BasePolicy):
    """Implementation of the Constrained Variational Policy Optimization (CVPO).

    (Docstring remains the same as before)
    """

    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        actor_optim: torch.optim.Optimizer,
        critic_optim: torch.optim.Optimizer,
        action_space: gym.Space,
        # CVPO specific arguments
        dist_fn: Type[torch.distributions.Distribution],
        max_episode_steps: int,
        logger: Optional[BaseLogger] = DummyLogger(),
        cost_limit: Union[List, float] = np.inf,
        tau: float = 0.05,
        gamma: float = 0.99,
        n_step: int = 2,
        # E-step
        estep_iter_num: int = 1,
        estep_kl: float = 0.02,
        estep_dual_max: float = 20,
        estep_dual_lr: float = 0.02,
        sample_act_num: int = 16,
        # M-step (General)
        mstep_iter_num: int = 1, # Usually 1, applies to actor update iterations
        # M-step (Continuous)
        mstep_kl_mu: float = 0.005,
        mstep_kl_std: float = 0.0005,
        mstep_dual_max: float = 0.5,
        mstep_dual_lr: float = 0.1,
        # M-step (Discrete)
        mstep_kl_discrete: float = 0.01,
        mstep_dual_lr_discrete: Optional[float] = None,
        mstep_dual_max_discrete: Optional[float] = None,
        # other param
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    ) -> None:

        # *** Call super().__init__() FIRST ***
        # BasePolicy should handle setting self.actor, self.critics, self.logger etc.
        super().__init__(
            actor=actor,
            critics=critics, # Pass the original critics list/module
            dist_fn=dist_fn,
            logger=logger or DummyLogger(), # Ensure logger is instantiated
            gamma=gamma,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            action_space=action_space,
            lr_scheduler=lr_scheduler,
            # Pass other args BasePolicy might need
        )

        # --- Now CVPO specific initializations ---

        # Optimizers are specific to CVPO's handling
        self.actor_optim = actor_optim
        self.critics_optim = critic_optim # Note: BasePolicy might have its own optimizers? Ensure consistency.

        # Create target networks (actor_old, critics_old)
        # Ensure self.actor and self.critics are set by super().__init__()
        if not hasattr(self, 'actor') or self.actor is None:
             raise ValueError("BasePolicy __init__ did not set self.actor.")
        if not hasattr(self, 'critics') or self.critics is None:
             raise ValueError("BasePolicy __init__ did not set self.critics.")

        self.actor_old = deepcopy(self.actor)
        self.actor_old.eval()
        self.critics_old = deepcopy(self.critics)
        self.critics_old.eval()


        # Determine device and dtype from actor parameters (now safe)
        try:
            self.device = next(self.actor.parameters()).device
            self.dtype = next(self.actor.parameters()).dtype
        except StopIteration:
             self.logger.warning("Actor has no parameters, defaulting device to CPU and dtype to float32.")
             self.device = torch.device("cpu")
             self.dtype = torch.float32
             # Move actor/critics manually if needed? Usually parameters determine this.
             self.actor.to(self.device, self.dtype)
             self.critics.to(self.device, self.dtype)
             self.actor_old.to(self.device, self.dtype)
             self.critics_old.to(self.device, self.dtype)


        # Determine action space type
        self._discrete = isinstance(self.action_space, gym.spaces.Discrete)
        if self._discrete:
             if not isinstance(self.dist_fn, type(torch.distributions.Categorical)):
                  self.logger.info("INFO: Using discrete action space, but dist_fn is not torch.distributions.Categorical (or cannot be checked).")
             if not hasattr(self.action_space, 'n'):
                 raise ValueError("Discrete action space missing 'n' attribute.")
             self._num_actions = self.action_space.n
        else: # Continuous
             try:
                 action_dim = self.action_space.shape[0] if hasattr(self.action_space, 'shape') else 1
                 dummy_loc = torch.zeros(action_dim, device=self.device)
                 dummy_scale = torch.ones(action_dim, device=self.device)
                 dist_instance = self.dist_fn(dummy_loc.unsqueeze(0), dummy_scale.unsqueeze(0))
                 if not isinstance(dist_instance, torch.distributions.Normal) and \
                    not isinstance(dist_instance, torch.distributions.Independent) and \
                    not (hasattr(dist_instance, 'base_dist') and isinstance(dist_instance.base_dist, torch.distributions.Normal)):
                     self.logger.info("INFO: Using continuous action space, but dist_fn does not seem to produce Normal or Independent(Normal(...)) distributions.")
             except Exception as e:
                 self.logger.info(f"INFO: Could not check if dist_fn produces Normal distributions for continuous space: {e}")


        # Cost limits setup
        # Ensure self.critics_num is set by BasePolicy or set it here
        if not hasattr(self, 'critics_num'):
             self.critics_num = len(self.critics)
        num_cost_critics = self.critics_num - 1
        if num_cost_critics < 0:
             raise ValueError("CVPO requires at least one critic (for reward).")

        if np.isscalar(cost_limit):
             cost_limit = [cost_limit] * num_cost_critics
        if len(cost_limit) != num_cost_critics:
             raise ValueError(f"Number of cost limits ({len(cost_limit)}) must match number of cost critics ({num_cost_critics})")
        self.cost_limit = cost_limit

        self.max_episode_steps = max_episode_steps
        # qc threshold in the E-step (threshold per step)
        # Use self.gamma if set by BasePolicy, otherwise use passed gamma
        current_gamma = getattr(self, 'gamma', gamma) # Prefer gamma set by BasePolicy if exists
        self.qc_thres = [
             c * (1 - current_gamma**self.max_episode_steps) / (1 - current_gamma) /
             self.max_episode_steps if (c != np.inf and current_gamma != 1.0 and self.max_episode_steps > 0) else np.inf
             for c in self.cost_limit
        ]
        self.logger.info("CVPO Step-wise Cost Thresholds (qc_thres): " + str(self.qc_thres))

        # E-step init
        self._estep_kl = estep_kl
        self._estep_iter_num = estep_iter_num
        self._estep_dual_max = estep_dual_max
        self._estep_dual_lr = estep_dual_lr
        self._sample_act_num = sample_act_num
        estep_dual_init = np.zeros(self.critics_num)
        estep_dual_init[0] = 1.0
        self.estep_dual = torch.tensor(
             estep_dual_init, requires_grad=True, device=self.device, dtype=self.dtype
        )
        self.estep_optim = torch.optim.Adam([self.estep_dual], lr=self._estep_dual_lr)

        # M-step init
        self._mstep_iter_num = mstep_iter_num
        if self._discrete:
            self._mstep_kl_target = mstep_kl_discrete
            self._mstep_dual_lr = mstep_dual_lr_discrete if mstep_dual_lr_discrete is not None else mstep_dual_lr
            self._mstep_dual_max = mstep_dual_max_discrete if mstep_dual_max_discrete is not None else mstep_dual_max
            self.mstep_dual_kl: Optional[torch.Tensor] = None
            self.mstep_dual_mu: Optional[torch.Tensor] = None
            self.mstep_dual_std: Optional[torch.Tensor] = None
        else: # Continuous
            self._mstep_kl_mu_target = mstep_kl_mu
            self._mstep_kl_std_target = mstep_kl_std
            self._mstep_dual_lr = mstep_dual_lr
            self._mstep_dual_max = mstep_dual_max
            self.mstep_dual_kl: Optional[torch.Tensor] = None
            self.mstep_dual_mu: Optional[torch.Tensor] = None
            self.mstep_dual_std: Optional[torch.Tensor] = None
        self.mstep_optim: Optional[torch.optim.Optimizer] = None

        # Other CVPO params
        self._estep_duration = 0
        self._mstep_duration = 0
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self.tau = tau
        self._n_step = n_step
        self.__eps = np.finfo(np.float32).eps.item() * 10

    # --- Rest of the methods (update_cost_limit, pre_update_fn, etc.) ---
    # --- remain the same as the previous version ---

    def update_cost_limit(self, cost_limit: Union[List, float]):
        """Update the cost limit threshold(s).

        :param Union[List, float] cost_limit: new cost threshold(s), matching number of cost critics.
        """
        num_cost_critics = self.critics_num - 1
        if np.isscalar(cost_limit):
             cost_limit = [cost_limit] * num_cost_critics
        if len(cost_limit) != num_cost_critics:
             raise ValueError(f"Number of cost limits ({len(cost_limit)}) must match number of cost critics ({num_cost_critics})")
        self.cost_limit = cost_limit

        current_gamma = getattr(self, 'gamma', 0.99) # Use stored gamma
        self.qc_thres = [
             c * (1 - current_gamma**self.max_episode_steps) / (1 - current_gamma) /
             self.max_episode_steps if (c != np.inf and current_gamma != 1.0 and self.max_episode_steps > 0) else np.inf
             for c in self.cost_limit
        ]
        self.logger.info("Updated CVPO Step-wise Cost Thresholds (qc_thres): " + str(self.qc_thres))


    def pre_update_fn(self, **kwarg: Any) -> Any:
        """Initialize the mstep optimizer and dual variables before learning."""
        mstep_dual_params = []
        if self._discrete:
            if self.mstep_dual_kl is None or self.mstep_dual_kl.device != self.device or self.mstep_dual_kl.dtype != self.dtype:
                self.mstep_dual_kl = torch.zeros(
                    1, requires_grad=True, device=self.device, dtype=self.dtype
                )
                self.logger.debug("Initialized M-step KL dual for discrete actions.")
            mstep_dual_params.append(self.mstep_dual_kl)
            self.mstep_dual_mu = None
            self.mstep_dual_std = None
        else: # Continuous
            if self.mstep_dual_mu is None or self.mstep_dual_mu.device != self.device or self.mstep_dual_mu.dtype != self.dtype:
                self.mstep_dual_mu = torch.zeros(
                    1, requires_grad=True, device=self.device, dtype=self.dtype
                )
                self.logger.debug("Initialized M-step mu KL dual for continuous actions.")
            if self.mstep_dual_std is None or self.mstep_dual_std.device != self.device or self.mstep_dual_std.dtype != self.dtype:
                self.mstep_dual_std = torch.zeros(
                    1, requires_grad=True, device=self.device, dtype=self.dtype
                )
                self.logger.debug("Initialized M-step std KL dual for continuous actions.")

            mstep_dual_params.extend([self.mstep_dual_mu, self.mstep_dual_std])
            self.mstep_dual_kl = None


        if mstep_dual_params:
             if self.mstep_optim is None or not self.mstep_optim.param_groups:
                 self.mstep_optim = torch.optim.Adam(mstep_dual_params, lr=self._mstep_dual_lr)
                 self.logger.debug(f"Initialized M-step dual optimizer with LR: {self._mstep_dual_lr}.")
             else:
                 # Ensure parameters in optimizer are correct (simple check by length)
                 current_optim_params = self.mstep_optim.param_groups[0]['params']
                 if len(current_optim_params) != len(mstep_dual_params) or \
                    any(p1 is not p2 for p1, p2 in zip(current_optim_params, mstep_dual_params)):
                      # Recreate if params differ significantly
                      self.mstep_optim = torch.optim.Adam(mstep_dual_params, lr=self._mstep_dual_lr)
                      self.logger.debug(f"Re-initialized M-step dual optimizer due to param changes.")
                 else:
                     # Update LR just in case it changed
                     for param_group in self.mstep_optim.param_groups:
                            param_group['lr'] = self._mstep_dual_lr

        else:
             self.mstep_optim = None
             self.logger.warning("No M-step dual parameters found to optimize!")

    def post_update_fn(self, **kwarg: Any) -> Any:
        """Update the old actor network after learning."""
        with torch.no_grad():
            self.actor_old.load_state_dict(self.actor.state_dict())
        # Commented out debug log
        # self.logger.debug("Updated actor_old network.")

    def train(self, mode: bool = True):
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critics.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target critic network."""
        # Use BasePolicy's soft_update if available
        if hasattr(super(), 'soft_update'):
            self.soft_update(self.critics_old, self.critics, self.tau)
        else: # Manual soft update
            with torch.no_grad():
                for o, n in zip(self.critics_old.parameters(), self.critics.parameters()):
                    o.data.copy_(o.data * (1.0 - self.tau) + n.data * self.tau)
            self.logger.debug("Manually performed soft-update on critics_old.")


    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> List[torch.Tensor]:
        """Compute target Q values for n-step returns."""
        batch = buffer[indices]
        batch.obs_next = to_torch_as(batch.obs_next, next(self.actor.parameters()))
        # Use current actor for next action
        obs_next_result = self(batch, model="actor", input='obs_next')
        act_next = obs_next_result.act

        target_q_list = []
        with torch.no_grad():
            for i in range(self.critics_num):
                if self._discrete:
                    act_next_proc = F.one_hot(act_next.long(), num_classes=self._num_actions).float()
                else:
                    # Use BasePolicy map_action if available
                     if hasattr(super(), 'map_action'):
                         act_next_proc = self.map_action(act_next)
                     else:
                         act_next_proc = act_next

                # Use predict method of critics_old (target critics)
                # Ensure predict method exists and handles inputs correctly
                if hasattr(self.critics_old[i], 'predict'):
                     target_q, _ = self.critics_old[i].predict(batch.obs_next, act_next_proc)
                else: # Fallback to direct call
                     target_q = self.critics_old[i](batch.obs_next, act_next_proc)
                     if isinstance(target_q, (list, tuple)): # Handle double Q manually if predict not available
                          target_q = torch.min(*target_q)

                target_q_list.append(target_q)

        return target_q_list

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Compute n-step returns for reward and costs."""
        current_gamma = getattr(self, 'gamma', 0.99) # Use stored gamma
        # Use BasePolicy's compute_nstep_returns if available
        if hasattr(super(), 'compute_nstep_returns'):
            batch = self.compute_nstep_returns(
                batch, buffer, indices, self._target_q, self._n_step
            )
        else:
            # Add basic n-step calculation if BasePolicy doesn't provide it (placeholder)
            self.logger.warning("BasePolicy compute_nstep_returns not found, n-step returns might be incorrect.")
            # Placeholder: requires manual implementation based on BasePolicy's expected behavior
            # batch.rets = batch.rew # Simplistic 1-step return if no n-step logic available
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        use_actor_old: bool = False,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data."""
        model_net = getattr(self, model) if not use_actor_old else self.actor_old
        obs = batch[input]
        obs = to_torch_as(obs, next(model_net.parameters()))
        logits, hidden = model_net(obs, state=state)

        if isinstance(logits, tuple):
             dist = self.dist_fn(*logits)
        elif isinstance(logits, list):
             dist = self.dist_fn(*logits)
        else:
             dist = self.dist_fn(logits)

        if self._deterministic_eval and not self.training:
            if self._discrete:
                 act = dist.probs.argmax(dim=-1) if hasattr(dist, 'probs') else dist.logits.argmax(dim=-1)
            else:
                if hasattr(dist, 'mean'): act = dist.mean
                elif hasattr(dist, 'mode'): act = dist.mode
                else: act = logits[0] if isinstance(logits, tuple) else logits
        else:
            act = dist.sample()

        return Batch(logits=logits, act=act, state=hidden, dist=dist)


    def critics_loss(
        self,
        batch: Batch, critics: Union[nn.Module, nn.ModuleList], optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, dict]:
        """Compute loss for the critic networks."""
        weight = getattr(batch, "weight", 1.0)
        total_critic_loss = 0
        stats_critic = {}
        td_list = []

        # Pass a parameter tensor from the first critic instead of the ModuleList
        act = to_torch_as(batch.act, next(critics[0].parameters()))
        obs = to_torch_as(batch.obs, next(critics[0].parameters()))

        if self._discrete:
             act_proc = F.one_hot(act.long(), num_classes=self._num_actions).float()
        else:
             if hasattr(super(), 'map_action'):
                 act_proc = self.map_action(act)
             else:
                 act_proc = act

        # Pass a parameter tensor from the first critic instead of the ModuleList
        rets = to_torch_as(batch.rets, next(critics[0].parameters()))
        if rets.ndim == 1:
             target_qs = rets.unsqueeze(1)
        elif rets.ndim == 2 and rets.shape[1] == self.critics_num:
             target_qs = rets
        elif rets.ndim == 3 and rets.shape[1] == self.critics_num and rets.shape[2] == 1:
             # Handle case where n_step_returns adds an extra dimension
             target_qs = rets.squeeze(-1)
        else:
             raise ValueError(f"batch.rets has unexpected shape: {rets.shape}, expected ({len(obs)}, {self.critics_num}) or ({len(obs)}, {self.critics_num}, 1)")


        for i in range(self.critics_num):
            target_q = target_qs[..., i].flatten()

            current_q_list = critics[i](obs, act_proc)
            if not isinstance(current_q_list, (list, tuple)):
                current_q_list = [current_q_list]

            loss_i = 0
            td_i_list = []
            valid_q_count = 0
            q_current_vals = [] # Store valid current Qs for logging mean
            for current_q in current_q_list:
                 if current_q is None: continue
                 current_q = current_q.flatten()
                 if current_q.shape != target_q.shape:
                     self.logger.warning(f"Shape mismatch in critic {i}: current_q {current_q.shape}, target_q {target_q.shape}")
                     continue
                 valid_q_count += 1
                 td = current_q - target_q.detach()
                 loss_i += (td.pow(2) * weight).mean()
                 td_i_list.append(td.detach())
                 q_current_vals.append(current_q.detach()) # Store detached value

            if valid_q_count > 0:
                loss_i /= valid_q_count
                total_critic_loss += loss_i
                mean_td_i = torch.mean(torch.stack(td_i_list, dim=0), dim=0)
                td_list.append(mean_td_i)
                stats_critic[f"loss/loss_q{i}"] = loss_i.item()
                stats_critic[f"value/q{i}_current_mean"] = torch.mean(torch.stack(q_current_vals)).item() if q_current_vals else 0.0
            else:
                 # Log zero loss/value if no valid heads, append zero TD
                 stats_critic[f"loss/loss_q{i}"] = 0.0
                 stats_critic[f"value/q{i}_current_mean"] = 0.0
                 td_list.append(torch.zeros_like(target_q))

            stats_critic[f"value/q{i}_target_mean"] = target_q.mean().item()
            if i >= 1:
                stats_critic[f"value/qc{i}_thres"] = self.qc_thres[i - 1] if i-1 < len(self.qc_thres) else np.inf


        if isinstance(total_critic_loss, torch.Tensor) and total_critic_loss.requires_grad:
            optimizer.zero_grad()
            total_critic_loss.backward()
            optimizer.step()
        elif total_critic_loss != 0:
             self.logger.warning("Critic loss computed but does not require grad.")


        batch_td_error = td_list[0] if td_list else torch.zeros_like(batch.obs[:, 0])

        stats_critic["loss/q_total"] = total_critic_loss.item() if isinstance(total_critic_loss, torch.Tensor) else total_critic_loss
        return batch_td_error, stats_critic


    def _estep_dual_loss(self, q_values_b_k_critics):
        """Compute the dual loss for the E-step optimization."""
        eta = self.estep_dual[0]
        lambdas = self.estep_dual[1:]
        eta = torch.clamp(eta, min=self.__eps)
        loss = eta * self._estep_kl

        combined_q = q_values_b_k_critics[0].detach()
        num_cost_critics = self.critics_num - 1
        for i in range(num_cost_critics):
            lambda_i = torch.clamp(lambdas[i], min=0.0)
            qc_i = q_values_b_k_critics[i + 1].detach()
            combined_q -= lambda_i * qc_i
            if i < len(self.qc_thres) and self.qc_thres[i] != np.inf:
                loss += lambda_i * self.qc_thres[i]

        K = q_values_b_k_critics[0].shape[1]
        if K == 0:
             logsumexp_mean = torch.tensor(0.0, device=eta.device, dtype=eta.dtype)
        else:
             logsumexp_term = torch.logsumexp(combined_q / eta, dim=1)
             logsumexp_mean = torch.mean(logsumexp_term - np.log(K))

        loss += eta * logsumexp_mean
        return loss

    @staticmethod
    def gaussian_kl(
        mu_old: torch.Tensor, std_old: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decoupled KL between two multivariate Gaussians with diagonal covariance."""
        std_old = torch.clamp_min(std_old, 1e-4)
        std = torch.clamp_min(std, 1e-4)
        var_old, var = std_old**2, std**2

        kl_mu = 0.5 * torch.sum((mu_old - mu)**2 / var_old, dim=-1)
        kl_mu = kl_mu.mean()

        kl_std = 0.5 * torch.sum(torch.log(var / var_old) + var_old / var - 1, dim=-1)
        kl_std = kl_std.mean()
        return kl_mu, kl_std


    def policy_loss(self, batch: Batch, **kwarg):
        """Compute policy loss, including E-step and M-step."""

        # ==================== E-step ====================
        t_start = time.time()
        K = self._sample_act_num
        batch_obs_estep = to_torch_as(batch.obs, self.estep_dual)
        B = batch_obs_estep.shape[0]
        if B == 0:
             self.logger.warning("policy_loss received empty batch, skipping update.")
             return

        with torch.no_grad():
             old_result = self(batch, input="obs", use_actor_old=True)
             old_dist = old_result.dist

             if self._discrete:
                  sample_act_indices = old_dist.sample((K,)) # (K, B)
                  sample_act_one_hot = F.one_hot(sample_act_indices, num_classes=self._num_actions).float() # (K, B, N)
                  expanded_obs = batch_obs_estep.unsqueeze(0).expand(K, -1, -1).reshape(K*B, -1)
                  critic_actions = sample_act_one_hot.reshape(K*B, -1)
             else:
                  sample_act_raw = old_dist.sample((K,)) # (K, B, da)
                  if hasattr(super(), 'map_action'):
                     critic_actions_mapped = self.map_action(sample_act_raw)
                  else:
                     critic_actions_mapped = sample_act_raw
                  expanded_obs = batch_obs_estep.unsqueeze(0).expand(K, -1, -1).reshape(K*B, -1)
                  critic_actions = critic_actions_mapped.reshape(K*B, -1)

             q_values_list = []
             for i in range(self.critics_num):
                  if hasattr(self.critics[i], 'predict'):
                      q_val, _ = self.critics[i].predict(expanded_obs, critic_actions)
                  else: # Fallback
                       q_val_raw = self.critics[i](expanded_obs, critic_actions)
                       q_val = torch.min(*q_val_raw) if isinstance(q_val_raw, (list, tuple)) else q_val_raw

                  q_val = q_val.reshape(K, B).T # (B, K)
                  q_values_list.append(q_val)

        q_values_detached = [q.detach() for q in q_values_list]
        for estep_iter in range(self._estep_iter_num):
            self.estep_optim.zero_grad()
            estep_loss = self._estep_dual_loss(q_values_detached)
            estep_loss.backward()
            self.estep_optim.step()
            if estep_iter == self._estep_iter_num - 1:
                 self.logger.store(tab="loss", estep_dual_loss=estep_loss.item())

        with torch.no_grad():
             self.estep_dual.data[0].clamp_(min=self.__eps, max=self._estep_dual_max)
             self.estep_dual.data[1:].clamp_(min=0.0, max=self._estep_dual_max)

        estep_dual_detached = self.estep_dual.detach()
        current_eta = estep_dual_detached[0].item()
        current_lambdas = estep_dual_detached[1:].cpu().numpy()
        self.logger.store(**{"estep/dual_eta": current_eta})
        for i, lam in enumerate(current_lambdas):
             self.logger.store(**{f"estep/dual_lambda{i+1}": lam})

        eta_opt = torch.clamp(estep_dual_detached[0], min=self.__eps)
        lambdas_opt = estep_dual_detached[1:]
        combined_q_opt = q_values_detached[0]
        for i in range(self.critics_num - 1):
            combined_q_opt -= lambdas_opt[i] * q_values_detached[i+1]

        optimal_q = torch.softmax(combined_q_opt / eta_opt, dim=1) # (B, K)
        optimal_q = optimal_q.T.detach() # (K, B)

        t_estep = time.time()
        self._estep_duration += t_estep - t_start
        self.logger.store(tab="time", estep_time_ms=(t_estep - t_start) * 1000)

        # ==================== M-step ====================
        if self.mstep_optim is None and (self._discrete or (self.mstep_dual_mu is not None and self.mstep_dual_std is not None)):
             self.logger.warning("M-step optimizer not initialized before policy_loss M-step. Attempting initialization.")
             self.pre_update_fn()

        for m_iter in range(self._mstep_iter_num):
            # Pass a parameter tensor instead of the module
            batch_obs_mstep = to_torch_as(batch.obs, next(self.actor.parameters()))
            result = self(batch, model="actor", input="obs")
            current_dist = result.dist

            loss_mle = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            if K > 0: # Avoid calculation if K=0 samples were taken
                if self._discrete:
                    log_likelihood = current_dist.log_prob(sample_act_indices) # (K, B)
                    loss_mle = -torch.sum(optimal_q * log_likelihood) / B
                else:
                    with torch.no_grad():
                        old_result_batch = self(batch, model="actor_old", input="obs")
                        mu_old, std_old = old_result_batch.logits
                        mu_old, std_old = mu_old.detach(), std_old.detach()
                    mu, std = result.logits

                    sample_act_flat = sample_act_raw.reshape(K * B, -1)
                    mu_flat = mu.repeat_interleave(K, 0)
                    std_flat = std.repeat_interleave(K, 0)
                    mu_old_flat = mu_old.repeat_interleave(K, 0)
                    std_old_flat = std_old.repeat_interleave(K, 0)

                    dist1 = self.dist_fn(mu_flat, std_old_flat)
                    dist2 = self.dist_fn(mu_old_flat, std_flat)
                    log_prob1 = dist1.log_prob(sample_act_flat).sum(-1)
                    log_prob2 = dist2.log_prob(sample_act_flat).sum(-1)
                    likelihood_flat = log_prob1 + log_prob2
                    likelihood = likelihood_flat.reshape(K, B)
                    loss_mle = -torch.sum(optimal_q * likelihood) / B

            loss_kl = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            kl_mu_val, kl_std_val = 0.0, 0.0
            kl_discrete_val = 0.0
            dual_mu_val, dual_std_val = 0.0, 0.0
            dual_kl_val = 0.0

            kl_div = None # Store KL div for actor loss

            if self.mstep_optim is not None:
                if self._discrete:
                    with torch.no_grad():
                         old_logits_batch, _ = self.actor_old(batch_obs_mstep)
                         old_dist_batch = self.dist_fn(old_logits_batch)
                    # Detach current_dist for KL calculation if current_dist requires grad?
                    # No, KL needs grad w.r.t current_dist for actor update.
                    kl_div = kl_divergence(old_dist_batch, current_dist).mean()
                    kl_discrete_val = kl_div.item()

                    if self.mstep_dual_kl is not None:
                         mstep_dual_loss = self.mstep_dual_kl * (self._mstep_kl_target - kl_div).detach()
                         self.mstep_optim.zero_grad()
                         mstep_dual_loss.backward() # Update dual only
                         self.mstep_optim.step()
                         self.mstep_dual_kl.data.clamp_(min=self.__eps, max=self._mstep_dual_max)
                         dual_kl_val = self.mstep_dual_kl.item()
                         # Actor loss uses non-detached dual, but needs grad through kl_div
                         loss_kl = dual_kl_val * (kl_div - self._mstep_kl_target)
                    else: pass # Warning already issued if None

                else: # Continuous
                    # Ensure mu_old etc are calculated if K=0 case skipped mle calc
                    with torch.no_grad():
                        old_result_batch = self(batch, model="actor_old", input="obs")
                        mu_old, std_old = old_result_batch.logits
                        mu_old, std_old = mu_old.detach(), std_old.detach()
                    mu, std = result.logits # Current params (already have from dist calc)

                    kl_mu, kl_std = self.gaussian_kl(mu_old, std_old, mu, std)
                    kl_mu_val, kl_std_val = kl_mu.item(), kl_std.item()

                    if self.mstep_dual_mu is not None and self.mstep_dual_std is not None:
                         mstep_dual_loss = self.mstep_dual_mu * (self._mstep_kl_mu_target - kl_mu).detach() \
                                       + self.mstep_dual_std * (self._mstep_kl_std_target - kl_std).detach()
                         self.mstep_optim.zero_grad()
                         mstep_dual_loss.backward() # Update duals only
                         self.mstep_optim.step()
                         self.mstep_dual_mu.data.clamp_(min=self.__eps, max=self._mstep_dual_max)
                         self.mstep_dual_std.data.clamp_(min=self.__eps, max=self._mstep_dual_max)
                         dual_mu_val = self.mstep_dual_mu.item()
                         dual_std_val = self.mstep_dual_std.item()
                         # Actor loss uses non-detached duals, needs grad through kl_mu/kl_std
                         loss_kl = dual_mu_val * (kl_mu - self._mstep_kl_mu_target) \
                                 + dual_std_val * (kl_std - self._mstep_kl_std_target)
                    else: pass # Warning already issued if None

            # Total actor loss
            loss_actor = loss_mle + loss_kl

            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            if self._mstep_iter_num == 1: break


        with torch.no_grad():
             # Recompute current_dist for entropy if needed (actor params changed)
             result = self(batch, model="actor", input="obs")
             current_dist = result.dist
             entropy = current_dist.entropy().mean().item()

        mstep_logs = {
             "loss/mstep_loss_mle": loss_mle.item(),
             "loss/mstep_loss_kl": loss_kl.item() if isinstance(loss_kl, torch.Tensor) else loss_kl,
             "loss/mstep_loss_total": loss_actor.item(),
             "value/entropy": entropy
        }
        if self._discrete:
             mstep_logs.update({"kl/mstep_kl_discrete": kl_discrete_val, "value/mstep_dual_kl": dual_kl_val})
        else:
              mstep_logs.update({"kl/mstep_kl_mu": kl_mu_val, "kl/mstep_kl_std": kl_std_val, "value/mstep_dual_mu": dual_mu_val, "value/mstep_dual_std": dual_std_val})
        self.logger.store(tab="mstep", **mstep_logs)

        t_mstep = time.time()
        self._mstep_duration += t_mstep - t_estep
        self.logger.store(tab="time", mstep_time_ms=(t_mstep - t_estep) * 1000)


    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Update policy parameters based on the batch data."""
        self.pre_update_fn() # Ensure duals/optimizer are ready

        td_error, stats_critic = self.critics_loss(batch, self.critics, self.critics_optim)
        self.logger.store(**stats_critic)
        batch.weight = td_error.abs() + self.__eps # Store abs TD error for PER

        self.policy_loss(batch) # Actor update (E+M steps)

        self.sync_weight() # Soft update target critics
        self.post_update_fn() # Hard update actor_old

        return self.logger.get_latest_scalars()


    def get_extra_state(self):
        """Save the dual variables and their optimizers' state_dict."""
        mstep_optim_state = None
        try:
            if self.mstep_optim is not None: mstep_optim_state = self.mstep_optim.state_dict()
        except Exception as e: self.logger.error(f"Error getting M-step optimizer state_dict: {e}")

        extra_state = {
            'estep_dual': self.estep_dual.detach().data.cpu(),
            'estep_optim': self.estep_optim.state_dict(),
            'mstep_dual_kl': self.mstep_dual_kl.detach().data.cpu() if self.mstep_dual_kl is not None else None,
            'mstep_dual_mu': self.mstep_dual_mu.detach().data.cpu() if self.mstep_dual_mu is not None else None,
            'mstep_dual_std': self.mstep_dual_std.detach().data.cpu() if self.mstep_dual_std is not None else None,
            'mstep_optim': mstep_optim_state,
        }
        return extra_state

    def set_extra_state(self, state):
        """Load the dual variables and their optimizers' state_dict."""
        try:
            self.estep_dual.data = state['estep_dual'].to(self.device, self.dtype)
            self.estep_optim.load_state_dict(state['estep_optim'])
        except Exception as e: self.logger.error(f"Error loading E-step duals/optimizer state: {e}")

        mstep_dual_params = []
        if state.get('mstep_dual_kl') is not None:
            if self.mstep_dual_kl is None: self.mstep_dual_kl = torch.zeros(1, requires_grad=True, device=self.device, dtype=self.dtype)
            self.mstep_dual_kl.data = state['mstep_dual_kl'].to(self.device, self.dtype)
            mstep_dual_params.append(self.mstep_dual_kl)
        else: self.mstep_dual_kl = None

        if state.get('mstep_dual_mu') is not None:
             if self.mstep_dual_mu is None: self.mstep_dual_mu = torch.zeros(1, requires_grad=True, device=self.device, dtype=self.dtype)
             self.mstep_dual_mu.data = state['mstep_dual_mu'].to(self.device, self.dtype)
             mstep_dual_params.append(self.mstep_dual_mu)
        else: self.mstep_dual_mu = None

        if state.get('mstep_dual_std') is not None:
             if self.mstep_dual_std is None: self.mstep_dual_std = torch.zeros(1, requires_grad=True, device=self.device, dtype=self.dtype)
             self.mstep_dual_std.data = state['mstep_dual_std'].to(self.device, self.dtype)
             mstep_dual_params.append(self.mstep_dual_std)
        else: self.mstep_dual_std = None

        if mstep_dual_params and state.get('mstep_optim') is not None:
             try:
                 if self.mstep_optim is None: # Create if doesn't exist
                      self.mstep_optim = torch.optim.Adam(mstep_dual_params, lr=self._mstep_dual_lr)
                 else: # Ensure params are updated in existing optimizer
                      self.mstep_optim.param_groups[0]['params'] = mstep_dual_params
                 self.mstep_optim.load_state_dict(state['mstep_optim'])
                 self.logger.info("Loaded M-step optimizer state.")
             except Exception as e: self.logger.error(f"Could not load M-step optimizer state: {e}")
        elif mstep_dual_params:
             self.mstep_optim = torch.optim.Adam(mstep_dual_params, lr=self._mstep_dual_lr)
             self.logger.info("Created fresh M-step optimizer as no state was found.")
        else: self.mstep_optim = None