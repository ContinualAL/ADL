# ADL
Autonomous Deep Learning: Continual Learning Approach for Dynamic Environments. Proceedings of the 2019 SIAM International Conference on Data Mining.

# Abstract
The feasibility of deep neural networks (DNNs) to address data stream problems still requires intensive study because of the static and offline nature of conventional deep learning approaches. A deep continual learning algorithm, namely autonomous deep learning (ADL), is proposed in this paper. Unlike traditional deep learning methods, ADL features a flexible structure where its network structure can be constructed from scratch with the absence of initial network structure via the self-constructing network structure. ADL specifically addresses catastrophic forgetting by having a different-depth structure which is capable of achieving a trade-off between plasticity and stability. Network significance (NS) formula is proposed to drive the hidden nodes growing and pruning mechanism. Drift detection scenario (DDS) is put forward to signal distributional changes in data streams which induce the creation of a new hidden layer. Maximum information compression index (MICI) method plays an important role as a complexity reduction module eliminating redundant layers. The efficacy of ADL is numerically validated under the prequential test-then-train procedure in lifelong environments using nine popular data stream problems. The numerical results demonstrate that ADL consistently outperforms recent continual learning methods while characterizing the automatic construction of network structures.

# Citation
If you use this code, please cite:

@INPROCEEDINGS{ADL,
author = {Andri Ashfahani and Mahardhika Pratama},
title = {Autonomous Deep Learning: Continual Learning Approach for Dynamic Environments},
booktitle = {In SIAM SDM},
year = {2019}
}
