# Applied AI course — reviewed bilingual edition

Open `index.html` for English or `zh-HK/index.html` for 香港繁體中文. The language switch returns to the corresponding page. Technical terms, code, formulas and English example sentences stay in English.

The course has 32 lecture readers, 32 tutorial pages and notebooks, weekly tests with worked solutions, semester examinations, a capstone, and linked primary references. Read lesson and Lesson slides are separate views. The 340 slides summarize key steps and link back to complete worked explanations. Reference details are collapsed by default, and inline diagrams and tables show the mechanisms directly.

## Publish to GitHub Pages

Extract the ZIP and upload all contents (including `assets`, `downloads`, `lectures`, `tutorials`, `tests`, `zh-HK`, and `.nojekyll`) to your repository's selected Pages source folder. In Settings → Pages select Deploy from a branch, then the intended branch and folder. Keep the same folder structure. The site uses relative paths and works under both a user-site root and a project subpath.

For custom domains, non-main branches or nested source folders, set the actual owner, repository, branch and path in `assets/site-config.js` for Colab links.

## Reading and editing

Search runs locally within the selected language. Progress and test notes are saved in browser localStorage and shared between languages on the same origin. They do not synchronize across devices. Reading the site requires no model API or code editor.

Editable source and one-command rebuild instructions are in `downloads/Course_Website_Source.zip`. English and Chinese pages and search indexes are derived together; no manual search-index edit is needed. Generated HTML is formatted for inspection.

## Validation

All 212 HTML pages, local links, anchors and JavaScript syntax were checked. All 32 notebooks and companion scripts passed syntax checks. Twenty-eight CPU-safe notebook paths, targeted week-26 DPO/distillation and week-28 invalid-JSON checks, and all 25 support tests passed. Training-memory slider behavior was exercised with a DOM test double. Reference outputs are included as clearly labeled comparison material. Actual pretrained-model downloads, GPU training/reload, CUDA and production-serving performance were not executed in this revision. No final visual browser check was available. Distributed executable notebook outputs remain empty.

## 香港繁體中文

按「閱讀講義」閱讀完整內容，或按「課堂 slides」查看重點步驟和圖解；兩種檢視分開顯示。較詳細的參考資料預設摺疊。網站提供香港繁體中文解說，保留 English technical terms、英文例句、公式和程式碼。可編輯 source 和 rebuild 說明在 downloads 的 source ZIP 內。
