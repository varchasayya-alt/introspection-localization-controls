# Third-party notices

## llama-introspection-new (Hahami et al.)

Studies 3–5 reuse material from <https://github.com/elyhahami18/llama-introspection-new> at commit
`7d9280d79e4aa37599dc0fd89974be4bc7a54827`. That repository accompanies *Detecting the
Disturbance* (arXiv 2512.12411).

| What | Where it appears here |
|---|---|
| The 100-sentence localization pool (`LOCALIZATION_SENTENCES`) | `stimuli/study3_stimuli.json` (`sentence_pool`, `draws`), `stimuli/study4_stimuli.json`, `stimuli/study5_stimuli.json`, and the per-trial result files |
| The simple and complex concept datasets and the baseline words | `stimuli/study3_stimuli.json` (`concepts`, `baseline_words_first50`, `complex_data`) |
| The localization prompt wording | `scripts/study3.py` (`INTRO3`, `Q_LOC3`, `PREFIX3`) |
| The 20 saved layer-2 concept vectors | **Not redistributed.** `scripts/download_their_vectors.py` fetches them from the original repository and checks their SHA-256 |

That material is used under the following licence:

```
MIT License

Copyright (c) 2026 Ely Hahami

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Model weights (not included)

No model weights are included. The studies used:
- `allenai/Llama-3.1-Tulu-3-8B` (revision `666943798adbde0b1aff34626007e26986a3c107`);
- `meta-llama/Llama-3.1-8B-Instruct` (revision `0e9e39f249a16976918f6564b8830bc894c89659`).

Both are governed by the Llama 3.1 Community License. Meta's model is gated on Hugging Face.
