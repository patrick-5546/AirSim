# Reinforcement Learning Examples

## Setup

1. Switch to this branch

    ```
    git switch user/patrick-5546/rl
    ```

2. Create and setup virtual environment

    ```
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -Ur requirements.txt
    ```

3. Download environments

    - Car: [latest Linux AirSimNeighborhood](https://github.com/microsoft/AirSim/releases/download/v1.8.1/AirSimNH.zip)
    - Drone: [latest Linux AirSimMountainLandscape with powerlines](https://github.com/microsoft/AirSim/releases/download/v1.2.0Linux/LandscapeMountains.zip)

## Run

1. Run environment: `./<path to environment>.sh`
2. Train model
    - Car: `python dqn_car.py`
    - Drone: `python dqn_drone.py`
