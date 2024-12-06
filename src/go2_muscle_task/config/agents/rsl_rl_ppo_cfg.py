from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class Go2MuscleRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 50
    experiment_name = "unitree_go2_muscle"
    empirical_normalization = False
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims = [512, 265, 128],
        critic_hidden_dims = [512, 265, 128],
        activation="elu"
    )
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.05,
        num_learning_epochs=6,
        num_mini_batches=10,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class MuscleAngleRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 50
    experiment_name = "unitree_go2_muscle_angle"
    empirical_normalization = False
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims = [128, 128, 128],
        critic_hidden_dims = [128, 128, 128],
        activation="elu"
    )
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.05,
        num_learning_epochs=6,
        num_mini_batches=10,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )