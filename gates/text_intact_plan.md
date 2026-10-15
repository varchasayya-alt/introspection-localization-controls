# Study 2 — text-intact search (PROPOSED, awaiting researcher approval)

**This is a new pre-registered question, not a rescue of Gate 2.** Gate 2's FAIL stands and is
reported as a finding: at the Gate 1 setting (layer 2, α 5), localization is explained by text
corruption.

**Question:** is there an injection setting where the text stays intact (a clean reader of the
copied text is at chance) but the injected model still localizes above chance?

Model, prompt, injection method, answer scoring, copy test and clean reader are **exactly as in
Gate 1** (`gates/gate1_plan.md`), so results are comparable.

---

## 1. What changes from Gate 1

**Weaker pushes and later layers.** Gate 1's winning push was about 2× the normal activity at
layer 2. Here strength is set *relative* to the normal activity size at each layer:

> relative strength r = α ÷ (median activation size at that layer, measured on practice set 0
> with nothing injected, before any injection run)

For reference: Gate 1's chosen setting was r ≈ 2.1; Phase A's layer 2 α 2 (r ≈ 0.84) still found
the sentence 14/30 times, but its text was not checked.

## 2. Practice step (calibration attempt 1)

| | |
|---|---|
| Layers | 2, 4, 8, 12, 16 (of 32) |
| Relative strength r | 0.25, 0.5, 1.0 |
| Settings | 5 × 3 = **15** |
| Material | Gate 1 practice material (sentence set 0; volcano, violin, octopus) × 10 positions = 30 trials per setting |
| Per setting | 30 injected localization trials + 30 copy-and-reader trials |

- **Text-intact** = clean reader ≤ 15%, i.e. **≤ 4 of 30**. Reader trials stop early once the
  reader reaches 5 correct (then the setting is "not intact"; the count so far is reported).
  The exact copy-back rate is reported for every setting.
- **Selection rule:** among text-intact settings, pick the one where the injected model scores
  highest; ties → lower r, then lower layer.
- **Practice pass:** selected setting has injected ≥ 8/30 (≥ 25%) **and** injected − reader
  ≥ 3/30 (≥ 10 points).
- **If it fails → attempt 2** (max 2 attempts, each logged before it runs). Default plan for
  attempt 2: same 5 layers with r ∈ {0.35, 0.7, 1.4} (fills the gaps and goes a bit stronger);
  the final attempt-2 grid is written in the calibration log after attempt 1 and before it runs.
- **If attempt 2 also fails → STOP** and write everything up.

## 3. Fresh test (only if practice passes)

- **New material, written and frozen before any run:** 5 new concepts and 4 new sentence sets
  (40 new everyday sentences on topics unrelated to the concepts). Nothing here was used in
  Gate 1.
- 5 concepts × 4 sets × 10 positions = **200 trials**. Every trial gets both an injected
  localization run and a copy-and-reader run, so the two are compared on the same trials.
- **PASS needs all three:**
  1. injected model ≥ 25% (≥ 50/200);
  2. injected − reader ≥ 10 points;
  3. reader ≤ 15% (≤ 30/200), so the setting is confirmed text-intact on the fresh trials too.
     *(This third condition is Claude's proposal: the researcher's rule says "at a text-intact
     setting", and this checks that on the fresh data. Needs approval.)*
- Also reported (descriptive, not pass/fail): exact copy-back rate, accuracy by position,
  answer distribution, no-injection answers for the new sets.
- Optional add-on (needs approval): 200 random-vector trials at the selected setting, to see
  whether any text-intact localization is concept-specific (~1 h).

## 4. The "never answers 1 or 2" bias

Checked in the Gate 1 data:
- **It is a default, not a broken prompt.** With nothing injected the model never names 1 or 2,
  and at layer 2 α 5 it never did in 300 trials. In Phase A it answered 1 or 2 only 23 times,
  and almost always correctly. The clean reader (11 times) and the typed-word condition (20
  times) also answer 1 or 2, again almost only when right. So the model needs strong evidence
  before it will say "1" or "2".
- **Effect on this search:** weak pushes will probably not overcome the default, so positions 1
  and 2 will likely score ~0 and the ceiling is effectively 80%. That doesn't threaten a 25% bar,
  and it affects the clean reader the same way, so the comparison stays fair. Chance stays 10%
  because the target cycles through all ten positions.
- **Reported:** accuracy on positions 1–2 vs 3–10, and the answer distribution, for every
  setting and for the fresh test. Primary rules use all 10 positions (as in Gate 1).

## 5. Run time (measured from Gate 1: injected trial ≈ 17 s, copy+reader trial ≈ 61 s)

| Step | Trials | Time |
|---|---|---|
| Sanity checks + measure activation sizes at layers 12 and 16 | — | ~15 min |
| Practice: injected trials | 15 × 30 = 450 | ~2.1 h |
| Practice: copy+reader trials | up to 450 (fewer with early stop) | ~4–7.6 h |
| Fresh test (only if practice passes) | 200 injected + 200 copy+reader | ~4.3 h |
| Optional random-vector add-on | 200 | ~1 h |
| **Total** | | **~11–15 h** (practice ≈ one night, test ≈ the next day) |

Attempt 2, if needed, adds roughly the same as one practice step.
