# Related Work Citation Map

## Citation Status Rules

Use the three strands below to organize Related Work. The anchor sources are supplied as verified research synthesis. The provisional section contains arXiv-only context and must never be described as peer reviewed. Do not use a local proposal citation unless its bibliographic metadata has been independently verified.

## 1. Predictive World Models and JEPA-Like Representations

| Citation | Venue/year/URL | Exact role | Claim-safe contrast |
|---|---|---|---|
| Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA) | CVPR 2023. https://arxiv.org/abs/2301.08243 | Establishes joint-embedding prediction as a representation-learning approach. | Do not claim I-JEPA studies action-conditioned physical rollouts, adaptive horizon selection, or symbolic abstraction. |
| Bardes et al., *V-JEPA: Self-Supervised Learning from Video with Joint-Embedding Predictive Architecture* | TMLR 2024. https://arxiv.org/abs/2404.08471 | Supplies a video JEPA reference for predictive representations. | Do not claim V-JEPA evaluates a joint horizon-description controller or provides engine-anchored physical labels. |
| Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination* (DreamerV1) | ICLR 2020. https://arxiv.org/abs/1912.01603 | Provides a latent world-model and imagined-control baseline family. | Do not claim DreamerV1 tests state-dependent predictive granularity rather than policy learning through latent imagination. |
| Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3) | Nature 2025. https://arxiv.org/abs/2301.04104 | Provides a later world-model reference for broad-domain latent control. | Do not claim DreamerV3 supplies the proposed relational supervision, shared endpoint evaluator, or evidence for BG-NS-JEPA. |

## 2. Temporal Abstraction, Adaptive Computation, and Multi-Resolution Control

| Citation | Venue/year/URL | Exact role | Claim-safe contrast |
|---|---|---|---|
| Sutton, Precup, and Singh, *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning* | AIJ 1999. https://doi.org/10.1016/S0004-3702(99)00052-1 | Defines the options framework for temporally extended action selection. | Do not claim options select a world model's predictive horizon and description mode. |
| Bacon, Harb, and Precup, *The Option-Critic Architecture* | AAAI 2017. https://arxiv.org/abs/1609.05140 | Provides a learned option-discovery and termination reference. | Do not claim Option-Critic establishes matched-compute predictive control or cross-mode world-model evaluation. |
| Sharma et al., *TempoRL: Learning When to Act* | ICML 2021. https://arxiv.org/abs/2106.05262 | Provides a temporal-abstraction baseline that varies an action repetition decision. | Do not claim TempoRL jointly changes representational abstraction or evaluates a predictive controller. |
| Snell et al., *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters* | ICLR 2025. https://arxiv.org/abs/2408.03314 | Provides context for explicit compute allocation and matched-compute comparisons. | Do not claim its compute-scaling results transfer directly to physical world models or validate the proposed controller. |

## 3. Physical Reasoning, Neurosymbolic Models, and Evaluation

| Citation | Venue/year/URL | Exact role | Claim-safe contrast |
|---|---|---|---|
| Bear et al., *Physion: Evaluating Physical Prediction from Vision in Humans and Machines* | NeurIPS Datasets and Benchmarks 2021. https://arxiv.org/abs/2106.08261 | Provides a physical-prediction benchmark reference. | Do not claim Physion provides NovPhy-style novelty adaptation, oracle supervision, or a BG-NS-JEPA result. |
| Yi et al., *CLEVRER: CoLlision Events for Video REpresentation and Reasoning* | ICLR 2020. https://arxiv.org/abs/1910.01442 | Provides a video reasoning benchmark with collision and counterfactual questions. | Do not claim CLEVRER tests joint predictive granularity or validates engine-anchored terminal outcomes. |
| Riochet et al., *IntPhys 2019: A Benchmark for Visual Intuitive Physics Reasoning* | TPAMI 2022. https://doi.org/10.1109/TPAMI.2021.3083839 | Provides an intuitive-physics benchmark reference. | Do not claim IntPhys establishes the proposed supervision hierarchy or controller evidence. |
| Bakhtin et al., *PHYRE: A New Benchmark for Physical Reasoning* | NeurIPS 2019. https://arxiv.org/abs/1908.05656 | Provides a task-oriented physical-reasoning benchmark reference. | Do not claim PHYRE evaluates adaptive world-model prediction modes. |
| Li et al., *I-PHYRE: Interactive Physical Reasoning* | ICLR 2024. https://openreview.net/forum?id=1bbPQShCT2 | Provides an interactive physical-reasoning benchmark reference. | Do not claim I-PHYRE supplies the required shared evaluator or establishes BG-NS-JEPA performance. |
| Yang et al., *CRAFT: A Benchmark for Causal Reasoning About Physical Systems from Videos* | Findings of ACL 2022. https://arxiv.org/abs/2012.04293 | Provides a causal physical-video reasoning benchmark reference. | Do not claim CRAFT validates a continuous-carrier design or relational controller. |
| Asai and Fukunaga, *Classical Planning in Deep Latent Space: Bridging the Subsymbolic-Symbolic Boundary* (Latplan) | AAAI 2018. https://arxiv.org/abs/1705.00154 | Optional appendix context only; omitted from the compact main-text section because planning is outside this manuscript's primary claim. | Do not use it as a main-text anchor or claim Latplan models persistent continuous physical cascades or supports the BG-NS-JEPA state-carrier claim. |
| Pinto, Gamage, Xue, Zhang, Nikonova, Stephenson, and Renz, “NovPhy: A physical reasoning benchmark for open-world AI systems” | *Artificial Intelligence*, vol. 336, p. 104198, 2024. https://doi.org/10.1016/j.artint.2024.104198 | Establishes the published NovPhy benchmark and its novelty detection/adaptation evaluation as the benchmark setting. | Do not present it as any BG-NS-JEPA result, enriched cohort, or controller evidence. |

## Provisional ArXiv-Only Context

The following items may inform a future draft only after metadata checking. Cite them as arXiv-only work. Do not describe them as peer reviewed, use them to establish venue policy, or use them as evidence for BG-NS-JEPA.

| Citation | Venue/year/URL | Exact role | Claim-safe contrast |
|---|---|---|---|
| Bardes et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* | arXiv 2025. https://arxiv.org/abs/2506.09985 | Provisional context for later JEPA-style video prediction. | Do not claim peer review, shared evaluation with the anchors, or controller support. |
| Ha and Schmidhuber, *World Models* | arXiv 2018. https://arxiv.org/abs/1803.10122 | Provisional historical context for learned latent environments. | Do not claim it addresses the proposed controller or modern matched-compute evaluation. |
| Graves, *Adaptive Computation Time for Recurrent Neural Networks* (ACT) | arXiv 2016. https://arxiv.org/abs/1603.08983 | Provisional context for adaptive allocation of recurrent computation. | Do not claim ACT jointly selects predictive time and representation. **TODO: verify** publication metadata before any non-arXiv citation. |
| Raposo et al., *Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models* | arXiv 2024. https://arxiv.org/abs/2404.02258 | Provisional context for conditional computation. | Do not claim language-model routing validates physical multiresolution control. |
| LeWorldModel, *Stable End-to-End Joint-Embedding Predictive Architecture from Pixels* | arXiv 2026. https://arxiv.org/abs/2603.19312 | Provisional 2026 JEPA context listed in the local proposal. | **TODO: verify** authors, version, and publication status. Do not treat local-proposal metadata as verified. |
| ThinkJEPA, *Empowering Latent World Models with Large Vision-Language Reasoning Model* | arXiv 2026. https://arxiv.org/abs/2603.22281 | Provisional 2026 JEPA context listed in the local proposal. | **TODO: verify** authors, version, and publication status. Do not treat local-proposal metadata as verified. |
| Causal-JEPA, *Learning World Models through Object-Level Latent Masking* | arXiv 2026. https://arxiv.org/abs/2602.11389 | Provisional 2026 JEPA context listed in the local proposal. | **TODO: verify** authors, version, and publication status. Do not treat local-proposal metadata as verified. |
| Sub-JEPA, *Subspace Gaussian Regularization for Stable End-to-End World Models* | arXiv 2026. https://arxiv.org/abs/2605.09241 | Provisional 2026 JEPA context listed in the local proposal. | **TODO: verify** authors, version, and publication status. Do not treat local-proposal metadata as verified. |
| Efficient World Models | **TODO: verify** precise title, authors, venue, year, and URL before use. | Potential context only if a verified citation is obtained. | Do not cite this placeholder or imply a comparison until metadata and relevance are verified. |

## Selected Three-Paragraph Outline

### 1. Predictive World Models and JEPA-Like Representations

I-JEPA and V-JEPA motivate prediction in representation space, while DreamerV1 and DreamerV3 motivate learned latent world models for control. These works establish relevant predictive-model families without resolving whether a model should adapt its horizon and description mode during a persistent physical cascade. The manuscript can contrast BG-NS-JEPA at the level of a specified mechanism: joint selection of the prediction problem over horizon and mode. It must not state that the mechanism improves prediction, planning, or physical plausibility until the common evaluator and central experiment are complete.

### 2. Temporal Abstraction, Adaptive Computation, and Multi-Resolution Control

Options, Option-Critic, and TempoRL motivate temporally extended decisions and learned timing. Snell et al. motivates explicit compute accounting. Together they identify useful control and efficiency questions, while leaving the proposed world-model decision distinct from action selection or language-model routing. The manuscript can define its proposed distinction as a controller over predictive horizon and description mode, with matched-compute controls. It must not claim that temporal-abstraction literature already proves joint representation control, or that the proposed controller is superior to single-axis and fixed controls before those comparisons exist.

### 3. Physical Reasoning, Neurosymbolic Models, and Evaluation

Physion, CLEVRER, IntPhys, PHYRE, I-PHYRE, and CRAFT provide complementary physical-reasoning and evaluation settings. Pinto et al. (2024) establishes NovPhy as the setting for novelty-aware physical reasoning rather than evidence for the proposed method. The manuscript can motivate engine-anchored supervision and a shared terminal-outcome evaluator as future evaluation design. It must not claim a released enriched cohort, a validated controller, reliable macro supervision, or BG-NS-JEPA results; Latplan remains appendix-only context if later needed.

## Suggested Reference Budget

Target 25-30 references. Start with the 15 verified anchors and Pinto et al. (2024). Reserve 5-7 references for direct mechanism precedents, 3-4 for evaluator or data-contract context, and at most two provisional arXiv citations after their metadata is checked. Omit unverified local-proposal citations from a submitted bibliography.

## Sentence-Level Do Not Claim Guide

| Do not write | Claim-safe alternative |
|---|---|
| “Prior work proves that joint temporal and symbolic adaptation is necessary.” | “Prior work motivates predictive representations, temporal abstraction, and physical reasoning, while the proposed joint mechanism remains to be tested.” |
| “BG-NS-JEPA outperforms fixed and factorized controllers.” | “The planned comparison tests whether BG-NS-JEPA improves a predeclared shared endpoint at matched compute.” |
| “NovPhy demonstrates the proposed controller's novelty adaptation.” | “NovPhy is the planned test setting for novelty-aware physical reasoning.” |
| “The enriched cohort provides oracle supervision for the study.” | “The enriched cohort and accepted oracle supervision remain blocked.” |
| “The kinetic analogy derives the controller.” | “The kinetic analogy is bounded intuition for the proposed state-dependent decision.” |
