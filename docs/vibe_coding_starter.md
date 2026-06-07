# Vibe Coding starter — apply dtSFM to your own target or library

dtSFM was built end-to-end through natural-language prompting (Claude Code) by a
domain expert with no Python programming experience. You can drive it the same
way. Paste the prompt below into an agentic coding assistant (e.g. Claude Code)
opened in a clone of this repository, fill in the bracketed fields, and let it
write and run the code.

---

## Prompt: score a drug against a target

```
I have this repository cloned (github.com/Reddy-BIIE-ETHZ/dtSFM) and the conda env
"calm" created from environment.yml. Using the production encoder weights
SFM-BIIE-ETHZ/dtSFM-v3 (encoder_b3_epoch010.pt) and the class
calm.encoder.model_v3.CALMEncoderV3:

1. Download the encoder checkpoint with huggingface_hub.
2. Encode this drug SMILES: [PASTE SMILES]
3. Encode this protein sequence (one-letter, no header): [PASTE SEQUENCE]
4. Print the cosine similarity (the binding-compatibility score).

Explain in plain language what the number means relative to a known binder.
```

## Prompt: rank a library for one target (repurposing)

```
Same repo and env. Using CALMEncoderV3 and the dtSFM-v3 encoder weights:

1. Read my library of SMILES from [PATH TO .smi / .csv, one column of SMILES].
2. Encode my target protein sequence: [PASTE SEQUENCE].
3. Encode every library molecule, compute cosine similarity to the target,
   and write a CSV ranked best-to-worst with columns: rank, smiles, cosine.
4. Show me the top 20 and a histogram of the score distribution.

Keep it memory-safe: batch the library and cache embeddings to disk.
```

## Prompt: generate novel molecules for a target (generative design)

```
Same repo and env. Using calm.decoder.model_dtsfm_v3.CALMDecoderV3 and the
dtSFM-v3 decoder weights (decoder_v02_step50000.pt):

1. Download the decoder checkpoint.
2. Run the pre-generation encoder sanity check
   (calm.decoder.decoder_smoketest_dtsfm_v3) for my target so I know whether the
   encoder recognises its chemotype: [PASTE SEQUENCE].
3. If the check passes, generate 200 candidate SMILES at temperature 0.8.
4. Run the 5-stage screen (calm.decoder.screen_decoder_dtsfm_v3): validity,
   dedup, drug-likeness, encoder-rerank, top-20.
5. Give me the top-20 SMILES with their drug-likeness and encoder rank.

Then tell me which ones are worth cofolding with AlphaFold-3.
```

---

**Reminder.** dtSFM is a *hypothesis generator*. Always confirm shortlisted
candidates with an orthogonal verifier (AlphaFold-3 — see
[`alphafold3_cofold_protocol.md`](alphafold3_cofold_protocol.md)) and, ultimately,
the wet lab. dtSFM cosine and AF3 confidence are uncorrelated by design; only
candidates that pass both are worth pursuing.
