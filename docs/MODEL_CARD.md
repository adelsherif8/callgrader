# Model card: CallGrader encoder

**Model.** ModernBERT-base with a 5-way linear head on mean-pooled token vectors, fine-tuned end to end.
Input: a business fact card + the call rendered as `AGENT:` / `CALLER:` / `TOOL` lines, first 192 + last 576 tokens.
Output: one probability per label (`passed`, `task_success`, `missed_handoff`, `hallucinated`, `injection_resisted`);
each label has its own threshold, tuned on validation calls.

**Intended use.** Triage. It decides which voice-agent calls a person should listen to first. It is not a final
judgement on a call, a person, or a business.

**Training data.** 1,260 simulated calls to three demo businesses from the VoiceDesk eval harness. Labels are silver
(code checks + an LLM judge), with known errors (see README, Failure analysis).

**Evaluation.** 5-fold cross-validation grouped by scenario, so test calls come from unseen scenarios. Majority-answer
and TF-IDF baselines are reported next to it. Results: README.

**Limits.**
- Three businesses, English only, simulated callers. Real calls have speech-to-text errors and crosstalk that are
  not in this data.
- Made-up facts can only be judged against the fact card. A business with no card gets weaker `hallucinated` scores.
- `missed_handoff` positives come from 5 scenarios; recall on new kinds of emergencies is uncertain.
- Calls longer than ~768 tokens lose their middle.

**Training cost.** About 6 minutes per epoch on an Apple M3 (8 GB), batch 1 with 16-step gradient accumulation and
gradient checkpointing. No paid compute.
