This directory includes a bunch of tests to check nccl vs. gloo performance.

```
tests/
├── README.md
├── send_recv_test.py
├── allreduce_test.py
├── pbs_run_tests.sh
```

`pbs_run_tests.sh` is a script that runs the tests on Derecho. 

To run the tests, you can use the following command:

```bash
qsub pbs_run_tests.sh
```

The timing averages are saved in the `benchmark_results.log` file. 

Example Output : 

```
------------------------------------
ENABLE_NCCL_OFI 0
nccl 2-18-6: send/recv warmup: 3.3412930965423584 sec, benchmark time: 0.003709876537322998 sec.
gloo       : send/recv warmup: 0.05069410800933838 sec, benchmark time: 0.005035305023193359 sec.
gloo 	   : broadcast warmup: 0.03912699222564697 sec, benchmark time: 0.0025135278701782227 sec
gloo 	   : all_reduce warmup: 0.006342172622680664 sec, benchmark time: 0.0052425146102905275 sec
ENABLE_NCCL_OFI 1
nccl 2-18-6: send/recv warmup: 3.7098723649978638 sec, benchmark time: 0.0016695857048034668 sec.
gloo       : send/recv warmup: 0.05840861797332764 sec, benchmark time: 0.004752683639526367 sec.
nccl 2-18-6: broadcast warmup: 1.0438247919082642 sec, benchmark time: 0.0001163482666015625 sec
nccl 2-18-6: all_reduce warmup: 0.0020563602447509766 sec, benchmark time: 0.0017300724983215332 sec
gloo 	   : broadcast warmup: 0.04237473011016846 sec, benchmark time: 0.0033962488174438476 sec
gloo 	   : all_reduce warmup: 0.006528019905090332 sec, benchmark time: 0.00484006404876709 sec
------------------------------------