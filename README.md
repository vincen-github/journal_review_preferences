# Journal Review Preferences

[English](README.md) | [简体中文](README.zh-CN.md)

**Give every journal an AI review assistant that understands its own evaluation standards.**

We aim to turn an institution’s historical manuscripts, reviews, and decisions into assistance it can customize, evaluate, and deploy privately. Institutions retain their standards, data, and model choice. The project is free and open source.

> “prices, like everything else, move along the line of least resistance.”[1]

AI lowers manuscript-production costs while expert attention remains scarce. General models offer useful analysis, but do not automatically reflect a venue’s standards. Here, resistance means time, resources, and workflow changes; momentum comes from each participant’s own benefit. When those interests align, gradual adoption becomes easier: **pre-submission feedback → editorial triage → human verification**.

- **Authors:** identify weaknesses and venue fit earlier, reducing misplaced submissions and resubmission loops; a future institution-controlled precheck could offer this guidance.
- **Reviewers:** focus attention on relevant work requiring deeper judgment.
- **Editors:** reduce screening and coordination burdens, locate disagreements, and keep valuable work in view.
- **Institutions:** expand capacity while preserving quality, reputation, distinctive standards, and control.

## Small interfaces, gradual adoption

- Import manuscript versions, human reviews, scores, decisions, and timestamps through a clear schema.
- Configure venue criteria, retrieve historical examples, and evaluate external score calibration with separate learning, validation, and test groups.
- Choose replaceable local/cloud adapters; local endpoints, model IDs, generation parameters, and structured responses are configurable. No base-model fine tuning is required.

The offline CLI and replaceable interfaces work; real rule-conditioned validation remains unfinished. [Implementation and experiment status](docs/EXPERIMENTS.md).

Score prediction has prior work, including [PeerRead](https://github.com/allenai/PeerRead/tree/master/code); our focus is configurable institutional interfaces, with gains still to be evaluated.

People may pursue model-specific shortcuts instead of improving research, motivating human oversight, manuscript-grounded evidence, and robustness checks.

Preference algorithms have a separate fit/freeze/conditioning interface; [prompt files](docs/PROMPT_INTERFACE.md) configure standards and output. See [the current score method](docs/EXPERIMENTS.md).

## Try it locally

With Python 3.10+, run from this folder:

```powershell
python -B -m journal_prefs demo --out my_first_demo
```

[Quick start](QUICKSTART_ZH.md) · [Technical guide](docs/TECHNICAL_GUIDE.md) · [Data pilot](docs/PUBLIC_DATA_PILOT.md) · [Verification](TEST_RESULTS.md) · [Chinese rationale](PROJECT_RATIONALE_ZH.md)

Code: [MIT](LICENSE). [Upstream contributions](THIRD_PARTY_NOTICES.md) are recorded separately. dot assisted concept development, research synthesis, coordination, and writing; Codex agents implemented and independently reviewed the prototype. [Contribution details](docs/TECHNICAL_GUIDE.md#project-contributions-and-ai-assistance).

We welcome use and improvement. If this project informs your work, we would appreciate an acknowledgment and a link to [this project](README.md). This voluntary request adds no license restriction.

[1] Lefèvre, Edwin. *Reminiscences of a Stock Operator*. New York: George H. Doran Company, 1923. A fictionalized account inspired by Jesse Livermore. [Catalog](https://catalog.hathitrust.org/Record/006583843).

Wensen Ma
