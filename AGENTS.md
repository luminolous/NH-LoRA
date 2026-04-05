# AGENTS.md

Repo goal:
This repository must remain the paper-faithful official implementation of NH-LoRA.

Global rules:
- Prioritize faithfulness to the attached NH-LoRA design paper over convenience.
- Do not keep large conceptual approximations if the paper specifies a clearer mechanism.
- Preserve public config/API where reasonably possible.
- Prefer minimal but correct changes.
- Add or update tests for every behavior-critical change.
- Do not claim full paper alignment if known gaps remain.

Validation:
- Run unit tests and available smoke tests after changes.
- If tests fail, report the failure clearly and fix if possible.

Documentation:
- Keep README and docs/paper_alignment.md updated whenever implementation semantics change.