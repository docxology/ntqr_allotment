# output/data/postdoc_votes/ — ntqr_allotment

Serialized per-vote cache for the live Gemma postdoc panel (e.g. `gemma3_4b_votes.json`), keyed on config hash, seed, reviewer, application, model digest, and decode params. Enables resumable, incremental reruns of `scripts/run_postdoc_panel.py --require-live`.
