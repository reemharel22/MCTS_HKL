from gym.envs.registration import registry, register, make, spec

# Algorithmic
# ----------------------------------------
register(
    id='hkl-v0',
    entry_point='HklEnv.envs.hkl:HklEnv',
)
register(
    id='hkl-v1',
    entry_point='HklEnv.envs.khkl:HklEnv',
)
