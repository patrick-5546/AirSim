# Runs

| Run Name              | Lambda | Commit                  | Changes    | Description                  | Problems    | Run Args                                                      |
| --------------------- | ------ | ----------------------- | ---------- | ---------------------------- | ----------- | ------------------------------------------------------------- |
| `2023-03-07_17-02-02` | 2      | Finetune constants      |            |                              | Errored out | `-a ppo`                                                      |
| `2023-03-07_18-34-52` | 1      | Finetune constants      |            | Set `learning_starts` to 10K | Can't turn  |                                                               |
| `2023-03-09_13-13-42` | 2      | Finetune constants      | Load model |                              | Can't turn  | `-a ppo -l drone_out/eval/2023-03-07_17-02-02/best_model.zip` |
| `2023-03-09_17-19-22` | 2      | Load model              |            | Don't set initial velocity   | Can't turn  | `-a ppo -l drone_out/eval/2023-03-07_17-02-02/best_model.zip` |
| `2023-03-09_17-41-32` | 1      | Load model              |            | Don't set initial velocity   | Can't turn  | `-l drone_out/eval/2023-03-07_18-34-52/best_model.zip`        |
| `2023-03-10_16-03-43` | 1      | Fix no initial velocity |            | Fix distance function        |             |                                                               |
| `2023-03-10_16-47-40` | 2      | Fix loading model       |            | Fix distance function        |             | `-a ppo -l drone_out/eval/2023-03-09_17-19-22/best_model.zip` |
