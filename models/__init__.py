from models.implicit_agent import ImplicitAgent


def make_agent(agent_cfg, envs):
    if agent_cfg["type"] == "implicit":
        return ImplicitAgent(envs, agent_cfg)
    raise ValueError(f"Unknown agent type {agent_cfg['type']}")