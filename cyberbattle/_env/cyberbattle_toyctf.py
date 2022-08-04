# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..samples.toyctf import toy_ctf
from . import cyberbattle_env


class CyberBattleToyCtf(cyberbattle_env.CyberBattleEnv):
    """CyberBattle simulation based on a toy CTF exercise"""

    def __init__(self, **kwargs):
        super().__init__(
            initial_environment=toy_ctf.new_environment(),
            **kwargs)


class CyberBattleToyCtfMultiAgent(cyberbattle_env.CyberBattleMultiAgentEnv):
    """CyberBattle simulation based on a toy CTF exercise"""

    def __init__(self, num_agents, size, **kwargs):
        super().__init__(
            num_attackers=num_agents,
            initial_environment=toy_ctf.new_multi_agent_environment(num_agents),
            **kwargs)
