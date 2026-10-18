# Study 6 report — Tülu-3-8B with the Study 4 arm-A protocol

Plan: `gates/study6_plan.md` (a8cf4f0). Predictions: `predictions.md` (9c602d9). Frozen setup:
4502395. Run 2026-10-17 20:27 → 2026-10-18 01:45, one attempt, no restarts, no deviations.
Full tables: `results/study6/summary.md`.

## Result in one sentence

With the authors' prompt (in Tülu's chat format), sentence-words-only injection and vectors
recomputed from Tülu, **Tülu-3-8B localizes the push with its text intact: main test PASS**. So
Tülu's earlier failures (Gate 1, Study 2) came from our old setup, not from the model.

## Numbers

| | Result |
|---|---|
| Sanity | all passed; clean copies exact in 20/20 draws (S7) |
| Arm A (concept vectors) | **186/200 = 93.0%** (95% CI 88.6–95.8%; chance 10%) |
| Arm A reader (clean model reading the copied text) | 28/200 = 14.0% |
| Exact-copy trials | **192/200**; injected 178/192 = 92.7%, reader 20/192 = 10.4% → **PASS** (needed ≥ 50% and ≤ 15%) |
| Arm C (random vectors, same size) | **186/200 = 93.0%**: identical to arm A → not concept-specific |
| Injection size | push 5 = 215% of Tülu's normal activation size at layer 2 vs 219% for Llama; ratio 0.985 (about the same) |
| Tülu vs the authors' Llama vectors | mean cosine 0.53: "avg" vectors 0.89 (0.76–1.00), "last" vectors 0.17 (0.07–0.24) |

Every concept scored 15–20 of 20 and every position 17–20 of 20. Unlike Study 1, Tülu answers
"1" and "2" normally.

## What it means

- **Conclusion (pre-set table): "setup."** In Study 1, our prompt and injection over whole
  sentences changed the text Tülu copied back (0/150 exact copies), and a clean reader could find
  the target from the text alone. With the Study 4 protocol, the text stays intact (192/200 exact
  copies), the reader is near chance, and Tülu still finds the pushed sentence 93% of the time.
- **Same picture as Llama (Study 4):** words-only push 93% vs Llama's 97.5%; random vectors work
  just as well (93% vs Llama's 91%). The test detects **where** processing was disturbed, not
  **what** concept was injected, in both models.
- The push was about the same relative size in both models, so the difference from Study 1 is not
  explained by a weaker or stronger push.
- The "last" vectors differ a lot between the two models while the "avg" vectors are similar; since
  random vectors work equally well, this does not change the result.

## Prediction scoring (rules a8cf4f0)

| # | Prediction | Outcome | Correct? |
|---|---|---|---|
| P1 | b (50–75%) | a (93%) | no |
| P2 | b (40–149 exact copies) | a (192) | no |
| P3 | b (FAIL, model < 50%) | a (PASS) | no |
| P4 | b (random clearly worse) | a (equal, 186 vs 186) | no |
| P5 | a (same relative size) | a (ratio 0.985) | **yes** |
| P6 | c (cosine < 0.5) | b (0.53) | no |

Score 1/6. Running total over Studies 1–6: 11/39.

## Limits

- One model checkpoint (Tülu final), one layer and strength, 20 draws.
- Tülu's chat format replaces the Llama headers both in the prompt and in the vector recipe (approved
  in advance); the recipe code itself was checked against the authors' files in Study 3.
- Because random vectors work as well as concept vectors, this is evidence that Tülu can locate a
  disturbance, not that it recognizes a concept.
