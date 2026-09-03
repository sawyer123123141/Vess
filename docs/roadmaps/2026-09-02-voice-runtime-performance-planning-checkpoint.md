# Voice runtime performance planning checkpoint — 2026-09-02

This is a planning checkpoint, not an implementation decision. It records the
real target-PC measurements that should drive the next voice-performance design
pass.

## Goal

Keep Vess fully local while making normal spoken interaction feel responsive
and keeping recognition and voice quality high. The immediate question is how
to fit and schedule the LLM, Whisper, Chatterbox Turbo, and their supporting
models on the target hardware without trading VRAM pressure for equally bad
model reload stalls.

Do not assume the answer is "move Whisper to CPU." Do not assume the answer is
"keep everything on GPU." Compare alternatives against end-to-end behavior.

## Target hardware

- Windows
- AMD Ryzen 7 5800X, 8C/16T
- NVIDIA RTX 3070, 8GB VRAM
- 16GB DDR4 system RAM, currently running at 2400 MT/s
- Ollama `qwen2.5:7b`, context 4096, keep-alive intended to stay warm

Historical baseline measurements in `CLAUDE.md` remain useful, but the current
voice experiments changed the set of plausible placements.

## Chatterbox installation findings

The first Chatterbox install had three independent dependency problems:

1. `torch==2.6.0+cpu` meant CUDA was unavailable.
2. `torchvision==0.28.0` was mismatched with Torch 2.6.0 and failed while
   registering `torchvision::nms`.
3. Reinstalling the matched CUDA Torch stack upgraded NumPy to 2.4.6, which was
   incompatible with Chatterbox 0.1.7 and the installed Numba. Restoring NumPy
   1.26.4 fixed that layer.

After those fixes, Chatterbox still failed while constructing
`perth.PerthImplicitWatermarker()` because Perth had swallowed an import failure
and exposed `PerthImplicitWatermarker = None`. The current Perth package still
imports deprecated `pkg_resources`; setuptools 84 triggered the failure.
Pinning setuptools to 80.9.0 restored the class and Chatterbox synthesis.

Known-good local stack:

- Python 3.11
- `chatterbox-tts==0.1.7`
- `numpy==1.26.4`
- `setuptools==80.9.0` / repository constraint `setuptools<81`
- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`
- CUDA visible on NVIDIA GeForce RTX 3070

The Perth `pkg_resources is deprecated` and diffusers
`LoRACompatibleLinear is deprecated` messages are warnings, not current runtime
failures. Do not "fix" those warnings by casually upgrading the working stack.

## Standalone Chatterbox Voice Lab measurements

Neutral benchmark, three runs per text. First short run includes model cold
load and took about 16.15s. Warm summary:

| text | median warm synthesis | median RTF |
| --- | ---: | ---: |
| short | 517.4 ms | 0.631 |
| medium | 903.7 ms | 0.486 |
| skeptical | 953.8 ms | 0.487 |
| question | 742.7 ms | 0.508 |
| emphatic | 697.7 ms | 0.506 |

Playful benchmark used `PerformanceCue(playful, 0.65)`, which currently appends
`[chuckle]`. First short run again included cold load and took about 16.16s.
Warm summary:

| text | median warm synthesis | median RTF |
| --- | ---: | ---: |
| short | 764.8 ms | 0.534 |
| medium | 1162.6 ms | 0.458 |
| skeptical | 1276.1 ms | 0.450 |
| question | 1104.0 ms | 0.446 |
| emphatic | 840.4 ms | 0.479 |

Conclusion: standalone Turbo is viable on the 3070. Playful `[chuckle]` costs
meaningful extra latency and should remain an A/B quality decision rather than
an automatic assumption.

## Integrated Vess cold interaction

Local runtime experiment changed the local config to Chatterbox Turbo and used
Whisper small on CUDA with `int8_float16`, beam size 5.

Wake / request:

- wake transcript: `Hey Vett`, accepted against `hey vess` at edit distance 2
- follow-up: `How was your day?`

Measured latency:

- Whisper cold model load: **31632.7 ms**
- endpoint wait: **457.9 ms**
- transcription: **133.6 ms**
- speech-to-transcript: **591.6 ms**
- LLM first clause: **618.1 ms**
- TTS worker wait: **52515.1 ms**
- first answer-clause synthesis: **3935.5 ms**
- first audio after the request: **56451.4 ms** from the TTS timing origin
- speech-end-to-playback: **57661.6 ms**

Critical event ordering:

1. The synthesis worker was preparing the acknowledgement `"Yeah?"`.
2. Chatterbox performed its cold model load inside that synchronous prepare.
3. A newer conversational generation arrived while that work still occupied the
   only synthesis worker.
4. When preparation finally completed, the acknowledgement generation was
   stale and correctly skipped.
5. Only then could the worker start synthesizing the actual answer.

This is not a 58-second LLM problem. It is a cold TTS / single-worker
head-of-line problem. Latest-generation invalidation correctly discards stale
results, but it cannot cancel Python while Chatterbox is still loading its model.

## Integrated warm interaction

Without restarting Vess, the next request was:

`Hey Vess, what's 2 plus 2?`

Measured latency:

- endpoint wait: **475.5 ms**
- transcription: **1077.5 ms**
- speech-to-transcript: **1553.2 ms**
- LLM first clause: **1360.6 ms**
- TTS worker wait: **0.1 ms**
- first clause: `Oh, that's an easy one!`
- first-clause performance: playful, intensity 0.65
- first TTS synthesis: **2477.9 ms**
- speech-end-to-playback: **5392.6 ms**
- second neutral clause synthesis: **1665.9 ms**
- playback gap between clauses: **0.2 ms**

This proves the 52.5-second stall is not steady state. It also shows that full
Vess makes Chatterbox substantially slower than its standalone benchmark.

The response itself was unnecessarily verbose for the request. `Four.` would be
better interaction design and cheaper to synthesize than two clauses ending in
`just like magic!`. Response-length policy is therefore part of performance,
not merely personality tuning.

## VRAM measurement after warm interaction

`nvidia-smi` while Vess remained running:

- **7854 MiB / 8192 MiB VRAM used**
- **3% GPU utilization** at the sampled instant
- 44 C, P8, 11 W
- compute processes included Ollama `llama-server.exe` and Vess Python
- Windows desktop / browser / ChatGPT processes also used GPU memory under WDDM

The exact WDDM per-process VRAM split was not exposed, but total occupancy is
enough to show that the 8GB card has almost no headroom.

Interpretation: current steady-state pressure is primarily **capacity**, not a
GPU compute unit pegged at 100%. This likely contributes to the integrated TTS
slowdown and creates risk of paging / allocation stalls / CUDA OOM as the system
changes state.

## Whisper findings

Earlier direct hardware testing rejected beam size 1 because it produced severe
transcription errors. Keeping the same model/device/settings and changing only
beam size to 5 produced the correct owner phrase:

`Say, plants turn sunlight into energy.`

That beam-5 transcription took 393.1 ms with RTF 0.14 in that isolated test.
Keep beam 5 as the accuracy baseline unless a different model/config is compared
with equivalent owner-speech quality data.

Integrated warm examples varied substantially: one transcription took 133.6ms,
one took 1077.5ms. Do not summarize CUDA Whisper as a single fixed latency from
two short samples.

Historical CPU small/int8 baseline was around 2.3x realtime. CPU Whisper is a
candidate only if total end-to-end latency and resource headroom improve enough
to justify that slower recognition path.

## Current architecture facts relevant to planning

- `VoiceOutput` has one synthesis worker and one playback worker.
- It reserves one prepared-audio slot so synthesis can run one clause ahead of
  playback.
- `prepare_acknowledgement("Yeah?")` currently enters the same synthesis worker
  queue as speech.
- If acknowledgement audio is missing when requested, it can be synthesized
  synchronously in that worker.
- Stale generation checks exist before, during supported Chatterbox generation
  boundaries, after synthesis, before prepared playback, and before playback.
- Existing cancellation hooks help obsolete **generation** work but cannot
  interrupt Chatterbox's initial model construction/loading.
- Chatterbox is lazy-loaded on first synthesis.
- Qwen is intended to remain warm via Ollama keep-alive.
- The face/render loop must never block while any of this happens.

## Planning questions

The next high-intelligence planning pass should inspect the actual current
source before proposing changes and answer these questions with evidence:

1. What is the best **resident model placement** for this exact 5800X / 3070
   8GB / 16GB machine?
   - Qwen GPU + Whisper GPU + Chatterbox GPU?
   - Qwen GPU + Whisper CPU + Chatterbox GPU?
   - smaller/different Whisper on GPU?
   - different quantization or placement for Qwen?
   - a different TTS configuration/model only if voice-quality evidence makes
     it worthwhile?
   - some staged residency policy?
2. What is the correct startup sequence?
   - serial warmup rather than simultaneous load?
   - when should Whisper, Qwen, and Chatterbox become ready?
   - should Vess accept speech before every heavyweight component is ready?
3. How should acknowledgement work?
   - pre-generated static WAV for `"Yeah?"`?
   - cache generated acknowledgement at startup?
   - separate lightweight acknowledgement voice?
   - no acknowledgement in some wake-with-query cases?
   Evaluate behavior and latency, not just implementation convenience.
4. Can Chatterbox model loading be separated from acknowledgement generation so
   cold initialization never monopolizes the only answer-synthesis worker?
5. Is one-ahead clause synthesis still the right shape under 8GB VRAM pressure?
   It currently achieves near-zero playback gaps once generation is underway.
6. How much latency is caused by VRAM capacity pressure versus text length,
   Chatterbox expressive tokens, concurrent Ollama residency, or other factors?
   Design experiments that can distinguish them.
7. What response-length policy should make trivial questions terse without
   harming normal conversation? Prefer a deterministic / prompt-level design
   over adding another model call.
8. Which changes should be configuration experiments first, and which deserve
   production code?
9. What measurements and acceptance thresholds prove an improvement rather than
   merely moving latency from one stage to another?
10. What failure modes appear when Windows itself consumes more VRAM, another
    app opens, or memory availability changes?

## Suggested acceptance targets to challenge, not blindly accept

These are initial targets for the planning discussion:

- warm speech endpoint + transcription: ideally under 1.0s for short owner
  utterances without materially reducing recognition accuracy
- warm LLM first clause: roughly 0.5-0.8s when the request is simple
- warm first TTS audio: roughly 1.0-1.5s after text is available for ordinary
  neutral clauses
- warm speech-end-to-audible response: roughly 2-3s for a simple request
- clause-to-clause gap remains effectively imperceptible
- cold startup should never create a 30-60s apparently-dead conversational
  interaction
- enough VRAM / RAM headroom that opening ordinary desktop applications does not
  unpredictably collapse throughput

A plan may reject these numbers if measurement or model constraints justify it.

## Constraints

- Fully local for now. No cloud model routing.
- Do not raise Qwen context above 4096 casually.
- Do not sacrifice owner-speech recognition quality just to produce a prettier
  latency number.
- Do not unload/reload heavyweight models per clause or per ordinary turn unless
  measured data proves the reload cost is acceptable.
- Do not block the render loop.
- Preserve latest-generation / stale-TTS safety.
- Preserve the one-clause-ahead playback behavior unless an alternative is
  demonstrably better.
- Keep implementation small and testable. Planning can be broad; the eventual
  implementation should still land one validated step at a time.

## Deliberately unresolved

No final decision has been made about:

- CPU vs GPU Whisper
- changing Whisper model size
- changing Qwen quantization or residency
- changing TTS model
- Chatterbox preload timing
- static vs generated acknowledgement
- how much playful `[chuckle]` should be used
- exact latency targets

The next session should resolve architecture from these measurements before
implementation begins.
