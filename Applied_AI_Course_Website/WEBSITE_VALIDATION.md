# Website validation — revised edition

106 HTML pages passed structural checks: one main heading per page, unique element IDs, and 5,790 local links/assets/anchors resolving. All 106 pages returned HTTP 200 from a loopback preview server. Six JavaScript files passed node --check.

32 lecture presentations contain 401 slides (8–23 per week), with the complete notes available separately on the same page. Browser checks covered next/previous, keyboard arrows, slide selection, final-slide disabled state, show-all view, presentation entry/exit and phone-sized layout. The mobile lecture test had no horizontal document overflow. Presentation entry was corrected to keep its header visible.

All 14 interactive lesson types initialized. The new prediction, token-ID, softmax, mixture and transformer-block controls were exercised, and no console errors were observed on the checked pages. The existing nine mechanisms retain their deterministic teaching simulations. Browser checks are representative, not an exhaustive accessibility audit or a claim that every browser/device was tested.

32 notebooks and scripts passed syntax checks, 27 CPU-capable tutorial paths executed, and 23 support-code tests passed. Model/CUDA execution limits are recorded in the bundled course VALIDATION.md.

Search and progress remain browser-local. GitHub Pages publication has not been performed; upload the extracted website contents according to README.md.
