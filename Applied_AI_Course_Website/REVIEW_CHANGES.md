# Follow-up review — 21 October 2026

This revision applies the supplied review within the existing 32-week course. English and Hong Kong Traditional Chinese prose are aligned; technical terms, examples, code and reference titles remain English where appropriate.

## A. Correctness and executable checks

- Tutorial/notebook27: two-dollar fare, three tickets, answer6; rejection tests fail if unsafe calculator expressions are accepted.
- Tutorial/notebook22: separate FP16 (50 tokens/s) and packed four-bit (200 tokens/s) bandwidth ceilings, with stated omissions.
- Supplement: corrected the 96 MiB example reference to week22.
- Checkpoint A2: supplied vocabulary, prefix-to-next-token targets; no premature shifted-batch requirement.
- Tests7/10: removed unrelated endings; test14 now distinguishes correctness from causal faithfulness.
- Tutorial13: decode continuation only; top-k/top-p and a two-step greedy counterexample; explicit cached/uncached attention equality.
- Tutorial14: explicit shift, supervised response-ending EOS.
- Tutorial11: nonidentical random values, causal invariance, unmasked negative control, source sensitivity and query-row independence.
- Tutorial12: every step logged, 20 steps on either device, explicit imports/UTF-8.
- Tutorial16: MAC counts and approximate FLOPs under stated exclusions.
- Tutorial18: persistence/filter check requires nonempty hits.
- Tutorial30: host allow-list tested independently of image enablement; hostile retrieved text exercises controller rejection.
- Notebook23: actual vector sizes, removed stray KV calculation, guarded PyTorch import and current profiler key.
- Early setup: tutorials01–16 need no coursekit. All tutorial pages retain full-pack links and explain when support files are needed.
- Tutorial28: confirmed parse failure is caught; trace now records invalid_json with planner_error terminal status. A CPU regression exercises it.

## B. Teaching and assessment improvements

- Week1: zero observations versus zero empirical probability, add-alpha smoothing, unseen-prefix executable case.
- Week5: pronoun example corrected; vanishing gradients, LSTM and GRU explained with a scalar derivative.
- Week7: row/column convention bridge and scope of H_kv clarified.
- Week9: verified the existing different-position RoPE calculation and same-relative-offset comparison; retained it.
- Week11: excluded source-only slide material; numerical Chinchilla ratio and 6ND example with historical scope.
- Week16: complete numeric forward trace with position additions, normalization, masked attention, both residuals, MLP, logits and probabilities.
- Week24: actual loss-mask list, JSONL/chat serialization explanation; concise reviewed answer targets include eligible distractors and preserve all12 records, including the cross-version task.
- Week28: MCP figure placed with the MCP introduction.
- Week31: integration and release decisions lead the lesson; the required review concerns integration/reproduction; adjacent extensions are optional. Amdahl main-text rounding uses2.11.
- Tutorial titles preserve CPU/GPU, CUDA, RAG, KV, LoRA, QLoRA and MLOps capitalization.
- Weeks17–32: explicit question-level marking points; expanded arithmetic steps and concrete final-week model answers/calculation.
- Masking questions distinguish zeroing without renormalization from the mathematically equivalent zero-then-renormalize operation.
- Clarified label order/indexing, selective-accuracy denominator, KV dimensions/factor2, column-vector LoRA multiplication/scaling, DPO reference log-ratios and API timing boundaries.
- Both semester exams retain100 points and now include the flagged core-topic gaps: recurrence, parameter counts, encoder-decoder visibility, perplexity, top-k/top-p, cosine, retrieval metrics/RRF/reranking, KV memory, CUDA, DPO and distillation.
- Every notebook/tutorial now includes reference output or explicit expected model/GPU checks. Prediction prompts precede answer-revealing code. Semicolon suites were expanded and imports exposed.
- Attention includes scaling and padding. Diffusion includes a schedule and imperfect-predictor error. Semantic-search query uses a paraphrase without bus. Retrieval demonstrates a genuine lexical miss and score-to-ranking-to-fusion progression.
- Context packing explicitly uses a strict rank-prefix policy, tested against a skipped-document fixture; real-model path follows the same policy.
- LoRA alpha/r scaling is visible. Saved/reloaded adapter checks compare logits under pinned base revision, rather than only finite loss.
- A CPU distillation gradient step shows teacher probabilities, student change, cross-entropy and KL.
- Local service uses an OS-selected port and preserves stderr.200 sequential local timings are labeled descriptive, not a production tail estimate.
- Final audits inspect submitted artifacts, malformed/empty files, required identities and40 checked cases. The12 supplied cases are explicitly smoke tests; the learner must author/check at least28 more.
- Replaced fragile private model-revision reads with public revision resolution; pinned the PyTorch/Transformers recipe; text IO uses UTF-8.

## Widgets and reading experience

- Training-memory worksheet responds to batch/tokens using a labeled partial saved-residual estimate, not an unsupported full-fit claim.
- Attention mask-off stage describes unrestricted keys; geometry reports current winners; diffusion cumulative signal stays positive; agent budget/display use four proposals; home title no longer repeats Applied AI.
- Tutorial instructions identify actual tasks; terms are defined at teaching boundaries; stale numeric lab references were removed.
- Both search indexes and the editable source archive are rebuilt from the same sources.

## Verification and limits

- 212 HTML pages and all local links/anchors/JavaScript syntax passed.
- 32 notebooks and companion scripts passed syntax checks.
- 28 CPU-safe paths and targeted DPO/distillation/invalid-JSON checks passed.
- 25 support tests passed, including new calculator and invalid-JSON regressions.
-Training-memory batch/token controls were exercised with a DOM test double.
-Pretrained model downloads, actual GPU training/reload, CUDA and production-serving behavior remain unexecuted. The source contains the checks for those paths. Visual browser verification was unavailable.

Primary API/scaling references checked during revision:
- https://huggingface.co/docs/transformers/main/models
- https://github.com/huggingface/transformers/blob/main/docs/source/en/quicktour.md
- https://docs.pytorch.org/docs/main/generated/torch.autograd.profiler_util.FunctionEventAvg.html
- https://arxiv.org/abs/2203.15556


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
