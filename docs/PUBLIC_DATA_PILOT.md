# Public data sources and boundaries

No real data is distributed with this repository. Importers are independently authored; downloaded source scripts are not executed. Users must establish permission for their own data and model transmission.

- AllenAI/PeerRead, commit `9bb37751781a900cee9e74ec3105997732c8e8e5`: https://github.com/allenai/PeerRead . Kang et al., *A Dataset of Peer Reviews (PeerRead)*, NAACL-HLT 2018, https://aclanthology.org/N18-1149/ . A local exploratory ICLR 2017 study used the author-released parsed texts and public reviews. No root license was verified at the pinned revision; the project MIT license does not cover these records. No redistribution permission is inferred.
- Daoze/ReviewRebuttal, revision `0171bacee24a9d405792a32231dabcdb3788b365`: https://huggingface.co/datasets/Daoze/ReviewRebuttal . Zhang et al., *Re2: A Consistency-ensured Dataset for Full-stage Peer Review and Multi-turn Rebuttal Discussions*, 2025, https://arxiv.org/abs/2505.07920 . Only a small metadata split was inspected locally, not the full manuscript archive. Its card declares Apache-2.0, which does not replace underlying article/review rights.

Parsed manuscript snapshots and review stages were not certified as pre-review/initial versions. Same-year paper holdout does not simulate deployment chronology or remove possible model-pretraining contamination. Missing scores are not zero, and all reviews/versions of a manuscript must stay in one group. Existing public labels do not make acceptance probability estimates valid.

Local compatible-model tests forward configurable endpoint/model/generation parameters using fake transports only. Ollama/vLLM-style compatibility is an intended path, not a tested deployment. See [technical guide](TECHNICAL_GUIDE.md) and [experiment status](EXPERIMENTS.md).
