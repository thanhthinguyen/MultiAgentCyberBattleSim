# MultiAgentCyberBattleSim

**MultiAgentCyberBattleSim** is an extension of Microsoft's [CyberBattleSim](https://github.com/microsoft/CyberBattleSim), designed to support **multi-agent cyber security simulation and experimentation**.

This codebase **inherits and builds upon the original CyberBattleSim framework**, extending its capabilities to enable research and experimentation involving multiple interacting agents in simulated cyber environments.

## Relationship to CyberBattleSim

MultiAgentCyberBattleSim is developed as an extension of:

**Microsoft CyberBattleSim**  
https://github.com/microsoft/CyberBattleSim

The original CyberBattleSim provides the underlying cyber-range simulation framework, including the environment, network topology, vulnerabilities, actions, and attacker/defender simulation mechanisms.

This project reuses and extends these core components to provide additional functionality for **multi-agent cyber security scenarios**.

> **Important:** This project is based on and inherits functionality from Microsoft's CyberBattleSim. Please refer to the original repository for the underlying framework, documentation, licensing information, and acknowledgements.

## Key Features

- Extension of the CyberBattleSim simulation framework.
- Support for **multiple cyber agents** operating within the same simulated environment.
- Multi-agent interaction and decision-making.
- Suitable for experiments involving reinforcement learning and multi-agent reinforcement learning.
- Provides a basis for evaluating different agent behaviours and strategies in simulated cyber environments.

## Project Structure

The project extends the original CyberBattleSim components while adding functionality required for multi-agent experiments.

Typical components may include:

```text
MultiAgentCyberBattleSim/
├── cyberbattle/
│   ├── ...                 # Extended CyberBattleSim components
│   └── ...
├── examples/
│   └── ...                 # Example experiments
├── tests/
│   └── ...                 # Tests
├── requirements.txt
└── README.md
```

The exact structure may evolve as the project develops.

## Installation

Clone the repository:

```bash
git clone <YOUR-REPOSITORY-URL>
cd MultiAgentCyberBattleSim
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Depending on the version of the codebase, additional installation or environment configuration may be required. Please refer to the relevant documentation and example scripts.

## Getting Started

After installation, users can run the provided examples to create and interact with multi-agent cyber security environments.

For example:

```bash
python <example_script>.py
```

Examples can be found in the `examples/` directory.

## Research Applications

MultiAgentCyberBattleSim is intended primarily as a **research and experimentation framework**. Potential applications include:

- Multi-agent reinforcement learning for cyber security
- Cooperative and competitive agent behaviours
- Cyber security decision-making
- Exploration–exploitation studies
- Evaluation of cyber security strategies
- Experimental comparison of single-agent and multi-agent approaches

## Relationship with Microsoft CyberBattleSim

This project should be considered an **extension rather than a replacement** for CyberBattleSim.

The underlying simulation concepts and components originate from Microsoft's CyberBattleSim. MultiAgentCyberBattleSim adds additional functionality on top of that framework to facilitate multi-agent research.

For the original project, documentation, and source code, please visit:

https://github.com/microsoft/CyberBattleSim

## Citation

If you use **MultiAgentCyberBattleSim** in academic research, please cite this repository and also acknowledge the original CyberBattleSim project.

Please refer to the Microsoft CyberBattleSim repository for its recommended citation:

https://github.com/microsoft/CyberBattleSim

## Acknowledgements

This project is built upon **Microsoft CyberBattleSim**.

We acknowledge Microsoft and the contributors to CyberBattleSim for developing the underlying cyber security simulation framework.

**Original project:**  
Microsoft CyberBattleSim  
https://github.com/microsoft/CyberBattleSim

## License

MultiAgentCyberBattleSim inherits from and builds upon CyberBattleSim. Please review the license of the original CyberBattleSim project and the license included in this repository before using, modifying, or redistributing the code.

See the original project for further information:

https://github.com/microsoft/CyberBattleSim
