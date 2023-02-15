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

4. Ensure that the first point of `pts` in `_compute_reward()` matches the car's starting point
    - The car's starting point is defined in `~/Documents/AirSim/settings.json`

    ```json
    {
      "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
      "SettingsVersion": 1.2,
      "Vehicles": {
        "PhysXCar": {
          "VehicleType": "PhysXCar",
          "DefaultVehicleState": "",
          "AutoCreate": true,
          "PawnPath": "",
          "EnableCollisionPassthrogh": false,
          "EnableCollisions": true,
          "RC": {
            "RemoteControlID": -1
          },
          "X": 0, "Y": -1, "Z": 0,
          "Pitch": 0, "Roll": 0, "Yaw": 0
        }
      }
    }
    ```

## Run

1. Run environment: `./<path to environment>.sh`
2. Train model
    - Car: `python dqn_car.py`
    - Drone: `python dqn_drone.py`
