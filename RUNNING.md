# Run locally

Python 3.10 or newer is sufficient. The prototype uses the Python standard library; no package installation, model download, account or API key is needed for the synthetic demo.

```powershell
python -B -m journal_prefs demo --out my_first_demo
python -B -m unittest discover -s tests -v
```

Run both commands from the repository root. Outputs from the demo are deliberately synthetic and do not establish review quality. Existing output directories are not overwritten; choose a fresh name for a new run.

Start with [the schema](docs/DATA_SCHEMA.md), [synthetic history](examples/history.synthetic.jsonl), and [institution rubric](examples/rubric.json). Model, preference algorithm and prompt files are separate interfaces. See [technical guide](docs/TECHNICAL_GUIDE.md), [custom algorithm](examples/custom_preferences.py) and [prompt configuration](docs/PROMPT_INTERFACE.md).

`configs/offline.json` is the default. Local mode can call an already running compatible loopback model service; this repository neither installs nor launches one. Cloud mode is disabled by default and requires explicit endpoint/model configuration and per-run consent. It transmits data and may incur provider charges. No real endpoint compatibility is promised by mock-based transport tests.

Research tools under `tools/` require separately obtained, locally authorized data and explicit model-call consent. None of the real manuscripts, reviews, model responses or experiment directories is distributed. They are not needed to run the demo or tests. A configured local model and an externally hosted model have different privacy boundaries.

Model fine tuning, production editorial integration and a public precheck service are not implemented.
