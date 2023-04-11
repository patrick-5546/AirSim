# Runs

| Run Name              | Lambda | Commit                         | Changes                       | Description                  | Problems        | Run Args                                                                          |
| --------------------- | ------ | ------------------------------ | ----------------------------- | ---------------------------- | --------------- | --------------------------------------------------------------------------------- |
| `2023-03-07_17-02-02` | 2      | Finetune constants             |                               |                              | Errored out     | `-a ppo -p nh_0 -d center`                                                        |
| `2023-03-07_18-34-52` | 1      | Finetune constants             |                               | Set `learning_starts` to 10K | Can't turn      | `-a dqn -p nh_0 -d center`                                                        |
| `2023-03-09_13-13-42` | 2      | Finetune constants             | Load model                    |                              | Can't turn      | `-a ppo -p nh_0 -d center -l drone_out/eval/2023-03-07_17-02-02/best_model.zip`   |
| `2023-03-09_17-19-22` | 2      | Load model                     |                               | Don't set initial velocity   | Going backwards | `-a ppo -p nh_0 -d center -l drone_out/eval/2023-03-07_17-02-02/best_model.zip`   |
| `2023-03-09_17-41-32` | 1      | Load model                     |                               | Don't set initial velocity   | Can't turn      | `-a dqn -p nh_0 -d center -l drone_out/eval/2023-03-07_18-34-52/best_model.zip`   |
| `2023-03-10_16-03-43` | 1      | Fix no initial velocity        |                               | Fix distance function        | Inconsistent    | `-a dqn -p nh_0 -d center`                                                        |
| `2023-03-10_16-53-35` | 2      | Update print statements        |                               | Fix distance function        | Going backwards | `-a ppo -p nh_0 -d center -l drone_out/eval/2023-03-09_17-19-22/best_model.zip`   |
| `2023-03-11_15-37-42` | 2      | Update print statements        | Update DIST_B from 0.2 to 0.3 | Fix distance function        |                 | `-a ppo -p nh_0 -d center`                                                        |
| `2023-03-13_16-38-19` | 2      | Update print statements        | Update DIST_B from 0.2 to 0.3 | Restart visualizer           |                 | `-a ppo -p nh_0 -d center -l drone_out/eval/2023-03-11_15-37-42/best_model.zip`   |
| `2023-03-14_13-51-59` | 1      | Update DIST_B from 0.2 to 0.3  |                               | Loaded @30k ts               |                 | `-a ppo -p nh_0 -d center -l drone_out/eval/2023-03-13_16-38-19/best_model.zip`   |
| `2023-03-18_07-51-48` | 1      | Add support for multiple paths |                               | Loaded @30k ts               |                 | `-a ppo -p lm_0 -d center -l drone_out/eval/2023-03-14_13-51-59/best_model.zip`   |
| `2023-03-21_11-29-39` | 2      | Add distance mode              |                               |                              | Bug fix @60k ts | `-a ppo -p nh_1 -d dest`                                                          |
| `2023-03-23_11-47-46` | 2      | Various fixes                  |                               |                              | Poor perf.      | `-a ppo -p nh_0 -d dest -l drone_out/eval/2023-03-21_11-29-39/best_model.zip`     |
| `2023-03-25_17-00-11` | 2      | Various fixes                  |                               |                              | `crash.log`     | `-a ppo -p lm_0 -d dest -l drone_out/eval/2023-03-23_11-47-46/best_model.zip`     |
| `2023-03-28_01-18-38` | 2      | Various fixes                  |                               |                              |                 | `-a ppo -p lm_0 -d dest -l drone_out/eval/2023-03-25_17-00-11/best_model.zip`     |
| `2023-04-03_09-43-32` | 2      | Add gps spoofing               |                               |                              | Training        | `-a ppo -p nh_0 -d dest -l drone_out/eval/2023-03-21_11-29-39/best_model.zip -s`  |
| `2023-04-05_13-55-37` | 2      | Refactor dqn_drone.py          |                               | Evaluate without training    |                 | `-a ppo -p nh_0 -d dest -l drone_out/eval/2023-03-21_11-29-39/best_model.zip -se` |

## Transfer Learning Evaluation

### Center Distance Mode Transfer Learning

- `2023-03-11_15-37-42`: training from scratch on `nh_0` path for 50k ts
<!-- - `2023-03-13_16-38-19`: restart visualizer to fix out-of-sync propellers and continue training for 250k ts
    - Can do straight line very well -->
- **Missing** `2023-03-14_13-51-59`: continue training on lambda 1 for 30k + 30k = 60k ts
- **Missing** `2023-03-18_07-51-48`: continue training on `lm_0` path
- Newer runs on lambda 1 are also missing

### Destination Distance Mode Transfer Learning

- `2023-03-21_11-29-39`: trained from scratch on `nh_1` path for 60k ts
- `2023-03-23_11-47-46`: fix bugs and continue training on `nh_0` path for 70k ts
- `2023-03-25_17-00-11`: continue training on `lm_0` path, which crashed for 14k ts
- `2023-03-28_01-18-38`: restart visualizer after crash and continue training

## GPS Spoofing Evaluation

- `2023-03-21_11-29-39`: model trained without gps spoofing: `mean_ep_length=26` and `mean_reward=200` at 60k ts
- `2023-04-05_13-55-37`: evaluating model on gps spoofing for 1000 episodes:
