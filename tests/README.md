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

