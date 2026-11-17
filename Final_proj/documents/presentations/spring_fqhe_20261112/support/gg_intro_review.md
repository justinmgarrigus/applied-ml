# GG-SPRING introduction review

Scientific review: **PASS**, 12 November 2026. Reviewed the new slide immediately before the N=7 geometric-sampling results against `geometry_sampling.py`, the GG branch in `train_geometry_edge.py`, and the saved `ne7_gg_spring_20261111/study.json` settings. No training or experimental-result files were changed.

The slide correctly presents GG as this project's experimental selection extension before the existing SPRING update. The physical energy objective remains fixed. It uses a Born candidate pool of M=128 configurations and a selected update batch of B=32.

Verified implementation points:

- Logarithmic derivative vectors describe response to neural-parameter changes. Geometric novelty refers to derivative information, not occupation-array distance or overlaps of orthogonal Slater basis states.
- Both the candidate configurations and the previous selected batch are scored at the current network parameters. The previous batch's stored configuration arrays are not reused as stale derivatives.
- Scores are centered with the pool mean and scaled using its diagonal quantum metric. The anchor span is removed with a ridge-smoothed projection; the footer properly identifies this approximation.
- The residual fraction is bounded between zero and one. The stated probability is the actual 20% uniform / 80% novelty mixture, including the 0.01 novelty floor.
- Selection is stochastic, with replacement and full support. Inverse index-probability weights proportional to 1/(M*pi_i) are correct for the original Born-pool selection stage.
- The footer properly retains the finite-batch bias of normalized ratio/covariance estimates and states that no ED labels enter selection.
- The slide describes the original static comparison against the previous selected batch. It does not confuse this with the later sequential within-batch variant or the beta=0.5 broader proposal.

The schematic and stated goal of complementary tangent information are appropriate. No deterministic farthest-point rule, exact full-QGT construction, or guaranteed convergence improvement is claimed. No unresolved scientific issue was found.
