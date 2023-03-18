# Runs

| Run Name              | Lambda | Commit                        | Changes                       | Description                  | Problems        | Run Args                                                      |
| --------------------- | ------ | ----------------------------- | ----------------------------- | ---------------------------- | --------------- | ------------------------------------------------------------- |
| `2023-03-07_17-02-02` | 2      | Finetune constants            |                               |                              | Errored out     | `-a ppo`                                                      |
| `2023-03-07_18-34-52` | 1      | Finetune constants            |                               | Set `learning_starts` to 10K | Can't turn      |                                                               |
| `2023-03-09_13-13-42` | 2      | Finetune constants            | Load model                    |                              | Can't turn      | `-a ppo -l drone_out/eval/2023-03-07_17-02-02/best_model.zip` |
| `2023-03-09_17-19-22` | 2      | Load model                    |                               | Don't set initial velocity   | Going backwards | `-a ppo -l drone_out/eval/2023-03-07_17-02-02/best_model.zip` |
| `2023-03-09_17-41-32` | 1      | Load model                    |                               | Don't set initial velocity   | Can't turn      | `-l drone_out/eval/2023-03-07_18-34-52/best_model.zip`        |
| `2023-03-10_16-03-43` | 1      | Fix no initial velocity       |                               | Fix distance function        | Inconsistent    |                                                               |
| `2023-03-10_16-53-35` | 2      | Update print statements       |                               | Fix distance function        | Going backwards | `-a ppo -l drone_out/eval/2023-03-09_17-19-22/best_model.zip` |
| `2023-03-11_15-37-42` | 2      | Update print statements       | Update DIST_B from 0.2 to 0.3 | Fix distance function        |                 | `-a ppo`                                                      |
| `2023-03-13_16-38-19` | 2      | Update print statements       | Update DIST_B from 0.2 to 0.3 | Restart visualizer           |                 | `-a ppo -l drone_out/eval/2023-03-11_15-37-42/best_model.zip` |
| `2023-03-14_13-51-59` | 1      | Update DIST_B from 0.2 to 0.3 |                               | Loaded at 30k timesteps      |                 | `-a ppo -l drone_out/eval/2023-03-13_16-38-19/best_model.zip` |
