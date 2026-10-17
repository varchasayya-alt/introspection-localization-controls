# Commit hash map (history rewrite before publication)

Before the repository was made public on 2026-10-17, its history was rewritten **once**. Every commit had this laptop's Windows home-folder path replaced with `%USERPROFILE%` (Windows-style paths) or `$HOME` (shell paths), as deviation D5. Author, committer, both dates, messages and commit order are unchanged, and no other content changed. This was verified commit by commit against the original history before publishing.

Rewriting changes every commit hash. The **current** versions of all documents cite the new hashes. **Older versions** inside the history still cite the old ones; use this table to translate.

| old hash | new hash | date (local) | commit |
|---|---|---|---|
| `0e71871` | `83447df` | 2026-10-14 23:31 | Gate 0: feasibility report, CPU timing, checkpoint ladders, control design |
| `acc7335` | `e268a7e` | 2026-10-15 00:46 | Gate 1 pre-registration: record researcher predictions and Gate 0 decisions |
| `0d82a56` | `aaa7ba8` | 2026-10-15 00:56 | Gate 1: freeze stimuli, plan details, model pin, and experiment code |
| `8ef360d` | `73e85a1` | 2026-10-15 01:49 | Gate 1: memory fix, enforced compute stop rule, crash-restart runner |
| `89965b9` | `2036c8a` | 2026-10-15 02:11 | Gate 1: add analysis script before results are known |
| `6b92130` | `8aa3e9b` | 2026-10-15 05:19 | Gate 1 Phase A: calibration attempt 1 passed (layer 2, alpha 5, 26/30) |
| `89f30e3` | `d27559a` | 2026-10-15 06:49 | Gate 1 Phase B: positive control passes (161/300 = 53.7%, chance 10%) |
| `ca120a2` | `48f22e3` | 2026-10-15 11:33 | Gate 1 passes, Gate 2 fails: localization explained by changed text |
| `a97bf82` | `da4892a` | 2026-10-15 16:27 | Study 2 (text-intact search): proposed design; note 1/2 answer bias |
| `f6eb6c3` | `229a20d` | 2026-10-15 22:08 | Study 2: write prediction scoring rules before recording predictions |
| `3bdd3f2` | `c5197b5` | 2026-10-15 22:08 | Study 2: record researcher predictions verbatim |
| `22e2cbd` | `a31372a` | 2026-10-15 22:19 | Study 2: approved plan, frozen fresh-test material, code and analysis |
| `e5caf91` | `34385fa` | 2026-10-16 07:02 | Study 2 practice attempt 1 passed: layer 8, r 0.5 (14/30 vs reader 3/30) |
| `faa4e18` | `fc47e50` | 2026-10-16 12:13 | Study 2: fresh test fails; no text-intact localization that holds up |
| `674f137` | `97b5003` | 2026-10-16 12:52 | Study 2 report: random-vector check of the move-off-favourite hint; Study 3 proposal |
| `c23d17a` | `b1b0336` | 2026-10-16 13:42 | Study 3: write prediction scoring rules before recording predictions |
| `d925dc9` | `a07886f` | 2026-10-16 13:42 | Study 3: record researcher predictions verbatim |
| `db5edbf` | `4e17e38` | 2026-10-16 13:49 | Study 3: approved plan, frozen stimuli, model pin, code and analysis |
| `de698e5` | `824e556` | 2026-10-16 16:04 | Log Study 4 hypothesis (label-digit push) during the Study 3 run |
| `385dfba` | `423b45f` | 2026-10-16 22:23 | Study 3: headline replicates (89%); text-corruption loophole not confirmed |
| `db69988` | `d84560b` | 2026-10-16 22:39 | Study 4 proposal (sentence-only vs label-only injection); log copy fix D3 |
| `ef2431c` | `ac653cb` | 2026-10-16 22:45 | Study 4: write prediction scoring rules before recording predictions |
| `c7a3dc6` | `15abe21` | 2026-10-16 22:45 | Study 4: record researcher predictions verbatim |
| `8d30992` | `da36436` | 2026-10-16 22:50 | Study 4: approved plan, frozen draws and vector hashes, code and analysis |
| `8d7d9ee` | `0c68fde` | 2026-10-17 07:29 | Study 4: sentence-only injection passes the text-intact test; label route also real |
| `1135cc3` | `2bef5d6` | 2026-10-17 14:06 | Study 5 proposal: question-specificity control (noticing vs spotlight) |
| `960ec22` | `ab0e61c` | 2026-10-17 14:15 | Study 5: write prediction scoring rules before recording predictions |
| `b948ceb` | `791e0e5` | 2026-10-17 14:15 | Study 5: record researcher predictions verbatim |
| `5a5b121` | `2134b4e` | 2026-10-17 14:18 | Study 5: approved plan, frozen draws, code and analysis |
| `ca4cb22` | `1930855` | 2026-10-17 18:56 | Study 5: push partly acts as a spotlight; overall "mixed" |
| `8861e8b` | `0db94ac` | 2026-10-17 18:56 | Study 5 report: complete the per-concept Q1 list |
| `7805231` | `b9d1596` | 2026-10-17 19:11 | Write-up and public-release preparation (no new runs) |
