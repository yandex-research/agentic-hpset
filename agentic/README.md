# `agentic/` — the module-generation instruction sets

The agentic module variants in `bin/<family>/modules/` were produced by an LLM
agent driven by the instruction sets in this directory — one per model family.
They document *how the results were produced*: the research loop, the evidence
bar every module had to clear, and the per-stage interface a new module must
mirror.

Each family directory is a self-contained set:

```
agentic/<family>/
  commands/autoresearch.md              # the orchestrator (the /autoresearch command)
  skills/autoresearch-inspect/          # read + internalize the pipeline (+ inspect_pipeline.py)
  skills/autoresearch-hypothesis/       # generate evidence-grounded ideas
  skills/autoresearch-implement/        # add + register one module
  skills/autoresearch-testing/          # smoke-test a module vs the all-v0 baseline
```

## The loop

1. **Inspect** the family's pipeline and list the live registries.
2. **Hypothesize** a set of module ideas, each stating *mechanism, support, gap,
   harm case* — no idea is implemented without named, established support and a
   specific gap the tuned baseline cannot cover. The bar is a ceiling, not a
   quota: fewer, stronger modules is the correct outcome.
3. **Implement** one vetted idea at a time as a new indexed module, leaving
   index 0 (the base reference) untouched.
4. **Test** each module on synthetic scenarios, keep/drop by correctness and
   efficiency vs the all-v0 baseline, then run the final interaction sweep.

The tuner then searches over the resulting module indices per dataset
(`bin/<family>/hparams.py`); a smoke-test pass is never treated as a quality
claim.
