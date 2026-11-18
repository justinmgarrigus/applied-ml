# Neural quantum states: optimizer comparison

Start with an **algorithm**, then choose **Ising** or **FQHE**. All three use
variational energy as the loss. They differ in the parameter update; all are
within the variational Monte Carlo (VMC) framework.

| Algorithm | Update | Start here |
|---|---|---|
| SPRING | Sample-space reconfiguration with remembered directions | [SPRING](SPRING/README.md) |
| VMC | Ordinary stochastic gradient descent on energy | [VMC](VMC/README.md) |
| MinSR | Memory-free, damped minimum-norm/SR control | [MinSR](MinSR/README.md) |

```text
Final_proj/
├── SPRING/                 # checked first, before adding the controls
│   ├── README.md
│   ├── Ising/
│   │   ├── PHYSICS.md
│   │   ├── RUN_TRAINING.md
│   │   ├── training/       # entry point, configurations, PARAMETERS.md
│   │   └── results/        # retained evidence and fresh run directories
│   └── FQHE/               # same structure
├── VMC/                    # Ising/ and FQHE/, same structure
├── MinSR/                  # Ising/ and FQHE/, same structure
├── shared/                 # common physics/model backend + licensed optimizer
├── benchmarks/             # shared ED solvers, inputs, outputs, validation
├── documents/              # proposal, papers and presentations
├── tools/                  # layout checks and curated Git export
└── AGENTS.md               # instructions for the next agent
```

There is **no benchmark folder inside a physics subfolder**. All methods use
the same [ED database](benchmarks/README.md). The frozen database has 11
Hamiltonian instances and 12,085 basis rows counted per instance: 1,488 Ising
and 10,597 FQHE. These are configurations, not supervised training labels.

Read [shared/README.md](shared/README.md) for installation. Each physics folder
explains the input, wavefunction output, Hamiltonian action, loss, sampling,
updates and saved files. Read its `training/PARAMETERS.md` before tuning.
Small smoke runs validate execution; they do not establish an optimizer
ranking. A matched, multi-seed convergence comparison remains research work.
K-FAC and Hubbard are not implemented here.

Historical results include the accurate small Ising example and the stopped
large FQHE study. The latter uses a separate angular-momentum-aware research
protocol, not the portable energy-only baseline. See
[SPRING/FQHE/results](SPRING/FQHE/results/README.md) for provenance and limits.
The local old physics-first trees remain frozen archives, preserving absolute
checkpoint paths and source hashes. New work starts in the algorithm folders;
the Git export excludes those large archives. See [migration](MIGRATION.md).

The [two-page proposal](documents/project_report/SPRING_Proposal.pdf) remains the
canonical proposal. The current course materials are under
[presentation](documents/presentation/) and [project report](documents/project_report/).
See [submission details](SUBMISSION.md) and
[third-party notices](THIRD_PARTY.md). Contributor: **PassengerAn**.
