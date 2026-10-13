"""RL configurations for the MPC-guided tasks."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def themis_ppo_runner_cfg(experiment_name: str = "themis_velocity") -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the MPC-guided velocity tasks."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      distribution_cfg={"class_name": "rsl_rl.modules.GaussianDistribution"},
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.01,
    ),
    experiment_name=experiment_name,
    max_iterations=10_000,
    save_interval=500,
  )
