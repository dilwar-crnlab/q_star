from gym.envs.registration import register

register(
    id="ModRMSA-v0",
    entry_point="optical_rl_gym.envs:ModifiedRMSAEnv",
)

register(
    id="ModDeepRMSA-v0",
    entry_point="optical_rl_gym.envs:ModifiedDeepRMSAEnv",
)
