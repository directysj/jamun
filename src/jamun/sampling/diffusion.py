from collections.abc import Callable

import numpy as np
import torch
from tqdm.auto import tqdm


def sigma_grid(sigma_min: float, sigma_max: float, rho: float, num_steps: int) -> torch.Tensor:
    step_indices = torch.arange(N)
    t_steps = (
        sigma_max ** (1 / rho) + (step_indices / (N - 1)) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    return t_steps


def heun_step(f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], y_hat: torch.Tensor, t_hat: torch.Tensor, t_next: torch.Tensor, use_second_order_correction: bool) -> torch.Tensor:
    # Euler step
    d_cur = f(y_hat, t_hat)  # -t_cur * score_fn(y_cur, t_cur)
    y_next = y_hat + (t_next - t_hat) * d_cur
    if use_second_order_correction:
        d_prime = f(y_next, t_next)  # -t_next * score_fn(y_next, t_next)
        y_next = y_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return y_next


def integrate_heun(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y_next: torch.Tensor,
    t_steps: torch.Tensor,
    S_churn: float,
    S_min: float,
    S_max: float,
    S_noise: float,
    num_steps: int,
    use_second_order_correction: bool,
    save_trajectory: bool,
    verbose: bool,
):
    traj = [] if save_trajectory else None
    if save_trajectory:
        traj.append(y_next)

    for i, (t_cur, t_next) in tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])), desc="heun", disable=not verbose):
        y_cur = y_next
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            y_hat = y_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(y_cur)
        else:
            t_hat = t_cur
            y_hat = y_cur
        y_next = heun_step(f, y_hat, t_hat, t_next, use_second_order_correction=(i < num_steps - 1) and use_second_order_correction)

        if save_trajectory:
            traj.append(y_next)

    if save_trajectory:
        traj = torch.stack(traj, dim=0)

    return y_next, traj


class DiffusionSampler:
    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        num_steps: int,
        y_init_distribution: torch.distributions.Distribution | None,
        verbose: bool,
        S_churn: float,
        S_min: float,
        S_max: float,
        S_noise: float,
        use_second_order_correction: bool,
        save_trajectory: bool,
        **kwargs,
    ):
        self.t_steps = sigma_grid(sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, num_steps=num_steps)
        self.sigma = sigma_max  # compat (used by ModelSamplingWrapper.sample_initial_noisy_positions)
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.num_t_steps = num_t_steps
        self.y_init_distribution = y_init_distribution
        self.verbose = verbose
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.save_trajectory = save_trajectory
        self.use_second_order_correction = use_second_order_correction

    def sample(
        self,
        model,
        batch_size: int | None = None,
        y_init: torch.Tensor | None = None,
        **kwargs,
    ):
        if y_init is None:
            if self.y_init_distribution is None:
                raise RuntimeError("either y_init and y_init_distribution must be supplied")
            y_init = self.y_init_distribution.sample(sample_shape=(batch_size,))

        def f(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return -t * model.score(y, t)

        y, y_traj = integrate_heun(
            f,
            y_init,
            self.t_steps,
            save_trajectory=self.save_trajectory,
            verbose=self.verbose,
            S_churn=self.S_churn,
            S_min=self.S_min,
            S_max=self.S_max,
            S_noise=self.S_noise,
            num_steps=self.num_steps,
            use_second_order_correction=self.use_second_order_correction,
        )

        return {
            "sample": y if y is not None else None,
            "xhat_traj": y_traj if y_traj is not None else None,
        }
