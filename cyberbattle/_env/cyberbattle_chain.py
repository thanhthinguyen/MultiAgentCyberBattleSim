# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CyberBattle environment based on a simple chain network structure"""

from ..samples.chainpattern import chainpattern
from . import cyberbattle_env


class CyberBattleChain(cyberbattle_env.CyberBattleEnv):
    """CyberBattle environment based on a simple chain network structure"""

    def __init__(self, size, **kwargs):
        self.size = size
        super().__init__(
            initial_environment=chainpattern.new_environment(size),
            **kwargs)

    @ property
    def name(self) -> str:
        return f"CyberBattleChain-{self.size}"


class CyberBattleChainMultiAgent(cyberbattle_env.CyberBattleMultiAgentEnv):
    """CyberBattle environment based on a simple chain network structure with multiple agents"""

    def __init__(self, num_agents, size, **kwargs):
        self.size = size
        super().__init__(
            num_attackers=num_agents,
            initial_environment=chainpattern.new_multi_agent_environment(num_agents, size),
            **kwargs)

    @ property
    def name(self) -> str:
        return f"CyberBattleChainMultiAgent-{self.size}"
