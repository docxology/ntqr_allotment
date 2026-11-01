# AGENTS.md — vendor/

Layout: `ntqr-0.9-py2.py3-none-any.whl` (vendored NTQR wheel) + `README.md`. Nothing else.
Invariant: pinned dependency — upgrade only with explicit user authorization; historical analysis artifacts retain their original provenance. Install path/config handled by project packaging.
Verify: wheel integrity via `unzip -t ntqr-0.9-py2.py3-none-any.whl` and the test suite.
- Repo-wide policy: `../AGENTS.md` (ongoing); the shared policy for template projects is maintained in the template repository, not restated here.
