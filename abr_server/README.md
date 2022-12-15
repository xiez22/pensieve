# ABR Algorithms
## Run
- `pip3 install -r requirements.txt`
- Execute:
```sh
python3 main.py -abr xxx
```
- `xxx` can be one of: `bb`, `rb`, `mpc`, `fastmpc`, `festive`, `pensieve`.

## LEGACY
## Testing

This part of code can test your model and compare it with other traditional ABR algorithm.

### To Test RL Model
1) Put testing data in folder `cooked_traces`
2) Put your model in folder `models`
3) Run `python3 get_video_size.py`
4) Modify the model path `NN_MODEL` in `test.py`
5) Run `python3 test.py`

### To Run Other ABR Algorithms
* Buffer-based (BB)

    `python3 bb.py`
    
* Model Predictive Control (MPC)

    `python3 mpc.py`
    
* Optimal (theoretically)

    `./dp.out`
    
### Plot the Results

1) Modify `SCHEMES` in `plot_results.py`
2) Run `python3 plot_results.py`
