# Runs

| Lambda | Run Name              | Commit                  | Changes | Description                  | Problems   | Run Args                                               |
| ------ | --------------------- | ----------------------- | ------- | ---------------------------- | ---------- | ------------------------------------------------------ |
| 1      | `2023-03-07_18-34-52` | Finetune Constants      |         | Set `learning_starts` to 10K | Can't turn |                                                        |
| 1      | `2023-03-09_17-41-32` | Load Model              |         | Don't set initial velocity   | Can't turn | `-l drone_out/eval/2023-03-07_18-34-52/best_model.zip` |
| 1      | `2023-03-10_16-03-43` | Fix no initial velocity |         | Fix distance function        |            |                                                        |
