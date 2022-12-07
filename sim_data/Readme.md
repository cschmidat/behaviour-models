# Data directory
This directory contains the validation accuracies of all the neural network models we analyzed.

- `onel.npz`: One-layer model with SL + Hebbian learning on v (Model 1) on isotropic input distribution
- `doublel_sw.npz`: Two-layer model with supervised learning on W and Hebbian learning on v (Model 2) on isotropic input distribution
- `doublel_sv.npz`: Two-layer model with Hebbian learning on W and SL on v (Model 3) on isotropic input distribution
- `doublel_sv_wrongpc.npz`: Two-layer model with Hebbian learning on W and SL on v (Model 3) on non-isotropic input distribution
- `doublel_sv_wrongpc_fsm.npz`: Two-layer model with similarity matching on W and SL on v (Model 4) on non-isotropic input distribution
- `doublel_svw_fsm.npz`: Two-layer model with similarity matching and SL on W and SL on v (Model 5) on non-aligned input distribution