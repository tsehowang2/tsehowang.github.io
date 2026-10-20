# Website validation

Created 20 October 2026. The complete course has been converted into 90 searchable HTML pages.

## Automated checks

All 90 HTML pages were parsed. Every page has one main h1 and unique element IDs. Local links, anchors, scripts, style sheets and download targets were checked; the final exact count is recorded by the packaging check. All four JavaScript assets passed Node syntax checks. Every HTML page returned HTTP 200 from the local static server. ZIP CRC and entry paths were checked after packaging.

## Browser checks in Microsoft Edge

The homepage and lecture layout were visually inspected. A 390-pixel mobile viewport was checked: all nine visual widgets fit within their containers and the document had no horizontal overflow.

Search for harness returned relevant lecture/tutorial results. Marking a lecture complete updated progress and survived a reload; the test mark was removed. A weekly answer note survived reload and was cleared after checking. A worked solution expanded and displayed the expected answer.

Attention: with the first query and causal mask, output was (2.000, 0.000); removing the mask changed it to (1.537, 1.483).

RAG: removing required evidence in a version comparison produced an explicit insufficiency path. Fusion: removing passage A gave Evidence Recall@2 = 1/2. Memory: the 500M-parameter, 16-byte full-training state example gave 7.45 GiB. LoRA: rank 64 gave 131072 trainable parameters for the 1024 by 1024 layer.

Agent: an unknown tool was rejected before execution; the never-ending planner stopped after three proposals with budget_exhausted. Diffusion at full noise showed signal coefficient 0 and noise coefficient 1. Generation stopped after its four designed continuation tokens. The geometry illustration displayed dot products .8 and 4 with equal cosine .8. Browser console error inspection returned no errors during these checks.

## Scope

Visual examples are teaching simulations, not real model execution. The original course pack separately documents which CPU, model and GPU labs were actually run in its authoring environment.

GitHub Pages publication is the remaining user deployment step; no destination repository was selected or changed. Colab link construction is configured for branch main and standard root hosting, with overrides documented for other layouts. Real GitHub deployment and Colab launch from that deployment have not been tested.

Progress and notes are browser-local and do not synchronize. No backend, analytics, login or code editor is included.

