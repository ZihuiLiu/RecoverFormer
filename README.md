# RecoverFormer

End-to-end contact-aware recovery for humanoid robots.

This repo contains the code, paper, and figures for the paper *"RecoverFormer:
End-to-End Contact-Aware Recovery for Humanoid Robots"*. RecoverFormer is a
single transformer policy that takes a 50-step observation history (proprioception
plus contact-region distances) and outputs 29-DoF joint targets for the Unitree G1
at 50 Hz. It combines three components: a causal transformer encoder, a latent
recovery mode head, and a contact affordance head — all trained end-to-end with PPO.

## Repository layout

```
.
├── main.tex, refs.bib, main.pdf   # paper source and compiled PDF
├── figs/                          # figures referenced by main.tex
└── code/
    ├── envs/g1_recovery_env.py    # MuJoCo G1 recovery environment
    ├── models/recoverformer.py    # transformer policy
    ├── models/baseline_mlp.py     # MLP baseline
    ├── train.py                   # PPO training entry point
    ├── evaluate.py                # episode rollout / RSR evaluation
    └── make_*.py                  # scripts that produce the paper figures
```

## Setup

```bash
conda create -n recoverformer python=3.10
conda activate recoverformer
pip install -r requirements.txt
```

You also need the Unitree G1 MuJoCo model files (`mujoco_menagerie/unitree_g1`).
The env loads them from a relative path; set `G1_ASSETS` if needed.

## Training

```bash
python code/train.py --env_type open_floor --total_steps 5000000
```

A full 5M-step run takes ~2 hours on a single RTX 5080. Checkpoints land in
`code/logs/<run_name>/`.

## Evaluation

```bash
python code/evaluate.py --ckpt code/logs/<run>/final.pt \
                        --env_type open_floor --force 150 --episodes 200
```

Reports Recovery Success Rate (RSR), time-to-stabilize, and peak torso tilt.

## Reproducing the paper figures

```bash
python code/make_architecture_fig.py    # Fig. 2  (architecture)
python code/make_robot_teaser.py        # Fig. 3  (rollout)
python code/make_balance_fig.py         # Fig. 4  (balance trajectories)
python code/make_mode_tsne_split.py     # Fig. 5  (latent mode t-SNE)
python code/make_hero_teaser.py         # Fig. 1  (hero teaser)
```

The latex paper compiles with:

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## License

MIT — see `LICENSE`.
