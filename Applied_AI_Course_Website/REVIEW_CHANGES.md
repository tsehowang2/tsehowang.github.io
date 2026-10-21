# Review changes — 20 October 2026

## Corrected details and explanations

- Unified the attention output rounding to [1.3395, 1.3210]; clarified cross-entropy rounding.
- Corrected the ANN wording and garbled updates sentence. ANN trades approximation and index design against latency/recall; HNSW can add memory overhead.
- Fixed “1, 257” and formatted the example token “to” clearly.
- Standardized d_model, d_head, d_value, B, T, H and H_kv, with a visible notation key. Memory uses GiB consistently and distinguishes decimal GB.
- Scoped the 12*d_model² parameter estimate; worked GQA and gated-MLP alternatives.
- Replaced the DPO beta=2 example with beta=0.1 and recomputed its loss. Explained that beta has no universal best range.
- Retained simple contrastive teaching logits and added a realistic cosine/temperature comparison.
- Added relative-offset RoPE, the NoPE qualification, unequal-frequency BPE, a genuine cosine/dot-product ranking reversal, cache ledgers, a nontrivial reasoning example and a numerical attention heatmap.

## Coverage added within the existing 32 lessons

- Tokenizer families, whitespace/digits and multilingual token budgets; embedding polysemy, anisotropy and analogy limits.
- Cross-entropy/MLE/KL, nats/bits, perplexity and stable log-softmax.
- Modern attention and architecture variants, scaling laws, state-space models, training schedules, weight decay, mixed precision and contamination.
- Decoding filters and speculative decoding; post-training PPO/GRPO/RLVR/constitutional ideas; reverse diffusion, latent diffusion, DiT, guidance, flow matching and VLMs.
- Embedding input prefixes, Matryoshka, ColBERT and SPLADE; HNSW parameters, IVF/PQ/quantization/DiskANN; concrete chunking strategies.
- Worked nDCG and McNemar, faithfulness/context precision, benchmark leakage and uncertainty.
- Quantized memory and bandwidth estimates, FSDP/ZeRO, warps/coalescing/tensor cores, cooperative launch, Triton and torch.compile.
- Explicit LoRA initialization, NF4/double quantization/paged optimizers, DoRA and rsLoRA.
- Native tool calling, structured outputs, dependencies/parallel calls, ReAct, coding/computer-use environments and state-based agent evaluation.
- vLLM/llama.cpp, continuous batching, PagedAttention and prefix caching.
- OWASP application risks, link/image exfiltration, jailbreaks, personal data/PDPO, fairness, licensing/copyright, energy and regulatory applicability.

## Resources and reading experience

- Updated Python support guidance and added multilingual E5-small and Qwen3-0.6B recipes. Older models remain labeled historical controls.
- Added FlashAttention-2/3, latent-diffusion and flow-matching references. Foundational papers remain useful historical/mechanism references.
- English and Hong Kong Traditional Chinese explanations are aligned; English technical terms, examples and code are retained.
- Reading and slide views are separate. Slides summarize mechanisms and link to complete sections. Exact duplicate prose is removed, and reference details are collapsed.
- Inline diagrams/tables, descriptive section navigation, reference titles and notebook links replace opaque identifiers.
- HTML is formatted; the editable source ZIP rebuilds pages and search indexes together. The source rebuild passed in a fresh directory.

## Verification limits

Static page/link/anchor/JavaScript checks, notebook syntax, 13 CPU tutorial paths, an independent DPO calculation, 23 support contract tests and worked arithmetic passed. Model downloads, GPU/training/CUDA/production-serving paths require separate runtime verification. A browser preview was unavailable in the final session, so this revision does not claim a completed visual/mobile browser check.
