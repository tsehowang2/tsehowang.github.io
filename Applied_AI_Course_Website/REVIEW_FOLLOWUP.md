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
