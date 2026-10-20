# English / 香港繁體中文 edition

Open `index.html`, then select **繁體中文** to read the corresponding Chinese page. You can also open `zh-HK/index.html` directly. The **English** link returns to the same page in English.

The Hong Kong Traditional Chinese edition covers the 32 lecture readers, all 401 HTML slides, 32 tutorials, weekly questions and worked solutions, examinations, project guidance, and interactive explanations. Technical and academic terms remain in English. Example words and sentences (such as “I go to school by bus”), code, equations, API names, filenames, and reading titles retain their original language. Downloadable notebooks and source materials are the original approved English files; their website explanations are translated.

Both editions are prebuilt static HTML and work offline. No translation service, account, API key, build step, or server is needed. Search operates within the selected language and includes retained English terminology. Browser progress and test notes are shared between languages on the same origin. English pages retain the approved teaching content.

Upload **all** extracted folders, including **zh-HK**, to the same GitHub Pages root. Both user sites and project sites use relative links. Keep **assets/site-config.js** configured as described below if you use a custom domain or nonstandard repository layout.

## 香港繁體中文版本

按頁面上方的「繁體中文」即可開啟同一頁的中文版本；按「English」可切回英文。解說使用香港繁體中文；token、logit、matrix、dot product、normalize、prompt、function 等 technical terms 保留英文。英文例句、程式碼及公式亦保留原文。兩種語言共用同一瀏覽器的進度和練習筆記。

---

# Applied AI course website

The complete course is readable on the website. No embedded code editor, API, package installation or build step is needed to read it. Open index.html locally, or host the extracted contents on GitHub Pages.

## Publish a new GitHub Pages project

1. Create a public GitHub repository, for example ai-course.
2. Extract the website ZIP on your computer.
3. Upload the contents of the extracted folder into the repository root on branch main. index.html, assets, lectures, tutorials, tests and downloads should be at the root. Upload the actual folders/files, not just the ZIP. Include the empty .nojekyll file.
4. Open the repository's Settings, then Pages.
5. Under Build and deployment, choose Deploy from a branch.
6. Choose main and /(root), then Save.
7. After GitHub's deployment finishes, use the Visit site link. A project repository named ai-course is normally published at https://YOUR_USERNAME.github.io/ai-course/.

Official instructions: https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

For a user homepage, use a repository named YOUR_USERNAME.github.io and upload these files at its root. If you already have a site there, use a separate project repository to avoid overwriting it.

## Optional Colab configuration

Notebook downloads work immediately. On a standard github.io user or project site with files on branch main at the repository root, Open in Colab links infer the owner/repository.

If using a custom domain, a different branch, a /docs source folder, or a nested folder inside a user site, set the four fields in assets/site-config.js:

- githubOwner: your GitHub username or organization.
- githubRepo: actual repository name.
- githubBranch: source branch, normally main.
- repositoryPath: folder prefix within that repository, such as docs or ai-course, without a leading/trailing slash. Leave empty for the root.

The notebook must exist at repositoryPath/downloads/notebooks/FILENAME.ipynb in the source repository. Outside a configured GitHub deployment, the Colab link opens Colab's homepage; download and upload the notebook manually.

## Content and reading features

32 lecture pages, 32 tutorial pages with readable Python, 32 weekly tests with 96 expandable solutions, two examinations with marking guides, project milestones/rubric, syllabus, primary readings, a worked supplement and the full revised course download.

Fourteen interactive visual lessons cover ordinary-sentence prediction, token IDs, softmax, weighted mixtures, transformer blocks, attention, embedding geometry, token generation, RAG, rank fusion, GPU state memory, LoRA, agent control and forward diffusion noising. The examples are deterministic teaching simulations, not running trained models or benchmark results.

Search runs locally in the browser. Lecture progress and working notes use browser localStorage and do not synchronize between devices. If browser storage is blocked, reading still works; saved progress/notes may not persist. Clear browser site data to reset them. There are no analytics, login, external font libraries or server calls.

Python and GPU exercises remain notebook/local tasks; the website does not execute them. The full revised course ZIP under downloads includes the source modules and fixtures used by the notebooks.

## Files and maintenance

index.html is the entry point. Paths are relative, so the site works under a project subpath. assets contains CSS, browser scripts and a search index. HTML content is prebuilt: edit an HTML page directly for a small correction, and update assets/search-index.js if its searchable text changes.

The notebooks, course Python scripts and course ZIP are downloads, not website server processes. Do not upload your private experiment logs or model checkpoints into this public site.

## Local preview

Double-click index.html for offline reading. For a local HTTP preview, run Python's standard library server from this directory:

    python -m http.server 8000 --bind 127.0.0.1

Then open http://127.0.0.1:8000/. No Python server is needed on GitHub Pages.

## Validation

See WEBSITE_VALIDATION.md for the actual website checks. Model-training and CUDA runtime checks are separately described in the revised course pack.

## Second-edition teaching sequence

Weeks 1–16 teach language modeling and transformers before applications. Weeks 17–32 cover retrieval, vector databases, RAG, evaluation, compute, CUDA, fine-tuning, tools, agents and delivery. Each lecture has a native HTML slide presentation with next/previous buttons, keyboard control, slide selection, presentation mode and an all-slides view, plus complete notes below. The notes develop an ordinary example, mechanism, worked reasoning and deeper study. Print reveals every slide. On mobile, long slides scroll vertically.

A revised syllabus, 32 individual tutorials, 96 questions with worked solutions and two aligned semester exams accompany the lectures. The first-semester exam contains no RAG or deployment prerequisite. The capstone starts in week 17.
