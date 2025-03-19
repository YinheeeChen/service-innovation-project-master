# Service Innovation Project

## Introduction
This repository contains various projects and implementations related to service innovation, including explanations of algorithms and reinforcement learning techniques.

## Directory Structure
- **explaining_algorithm/**: Contains explanations and implementations of various algorithms.
- **reinforcement_learning/**: Contains subdirectories for different reinforcement learning techniques:
  - **DQN/**: Implementation of Deep Q-Network.
  - **PPO/**: Implementation of Proximal Policy Optimization.
  - **SQN/**: Implementation of Soft Q-Network.

## Installation
To set up this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/YinheeeChen/service-innovation-project-master.git
   cd service-innovation-project-master
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Explaining Algorithm
Navigate to the `explaining_algorithm` directory to find various algorithm explanations and their implementations. Run the respective Python scripts to see the algorithms in action.

### Reinforcement Learning
Navigate to the `reinforcement_learning` directory and choose the desired subdirectory (DQN, PPO, SQN) to explore and run the implemented reinforcement learning techniques.

#### Example: Running DQN
1. Navigate to the DQN directory:
   ```sh
   cd reinforcement_learning/DQN
   ```

2. Run the DQN script:
   ```sh
   python dqn_main.py
   ```

## Project Structure
- **explaining_algorithm/**: Contains various algorithm explanations and implementations.
- **reinforcement_learning/**: Contains subdirectories for different reinforcement learning techniques.
  - **DQN/**: Contains the implementation of Deep Q-Network.
  - **PPO/**: Contains the implementation of Proximal Policy Optimization.
  - **SQN/**: Contains the implementation of Soft Q-Network.
- **requirements.txt**: Lists the dependencies required for the project.
