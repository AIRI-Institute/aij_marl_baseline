# Public Baseline for AIJ Multi-Agent RL Contest
## Description

This repository contains an [implementation](baseline_eng.ipynb) of 
[VDN](https://arxiv.org/abs/1706.05296) cooperative Multi-Agent RL algorithm.
The main assumption behind VDN is linear decomposition of common rewards, 
that is, the sum of the individual utilities of agents equals the common 
reward. Although this assumption limits us to cooperative strategies only, 
VDN still serves as a solid baseline for many MARL tasks.

This baseline implementation allows to get Mean Focal Score of approximately 
42 when submitted to testing system (Comparing to ~4 for a random policy).

## Required resources

Training this baseline solution takes approximately 3 hours with
GPU and requires ~15gb. of RAM and ~1gb. of GPU memory. To reduce RAM usage 
you can decrease replay buffer size in training config.

## Creating Baseline Submission

__Step 1:__ Install requirements with ```pip install -r requirements.txt``` command. 
Requirements file also includes [public repository](https://github.com/AIRI-Institute/aij_multiagent_rl)
with `aij_multiagent_rl` python module which contains environment and base agent class for the contest.
In case of problems with installation, consider installing package from source.

__Step 2:__ Iteratively run all cells in [implementation notebook](baseline_eng.ipynb).
The resulting artifact of running this notebook is `submission_vdn` directory.

__Step 3:__ Run provided tests in order to validate your submission.

1) Insert your submission directory path to [test config file](tests/test_config.yaml)
using `submission_dir` key. By default, its value is set to `"submission_vdn"`
so you don't have to change anything. However, you can change this path in order
to validate your custom submissions later on.
2) Run provided submission tests with ```pytest tests``` command from repository root path 

__Step 4:__ Pack submission as .zip archive and send it to testing system.
__Important note:__ submission files should be at the top level of the compressed archive:

Correct:
```
submission.zip
    ├── utils         # Directory with modules required for the agent class (optional)
    ├── model.py      # Script with implementation of agent(s) class and factory method
    └── agents        # Directory with agent artifacts (fixed name) 
```
Not correct:
```
submission.zip
    └── submission
        ├── utils         # Directory with modules required for the agent class (optional)
        ├── model.py      # Script with implementation of agent(s) class and factory method
        └── agents        # Directory with agent artifacts (fixed name) 
```

## Unit tests
This repository includes unit tests for checking submission correctness. Submission
that is generated after executing all cells of notebook attached should pass all tests
(and grading system) without errors. Running testing procedure is described above.

It should be noted that one of tests includes agent action sampling performance testing.
If the configuration of the machine on which tests are run is very different from the container resources 
configuration (see Limitations paragraph in the [description](https://dsworks.ru/en/champ/multiagent-ai))
warning could be ignored, otherwise you should pay attention to it.

__We strongly recommend to run tests for submission before sending it.__

## Docker
Repository also includes [Docker files](docker) which can be used to
reconstruct testing system's environment in which your submission will be run
as well as for local development.
