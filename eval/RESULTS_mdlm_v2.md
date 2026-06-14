# Sampler v2 diagnostic — re-eval of SAME 66.8M checkpoint

Цель: отделить decode-проблему от capacity/данных. Веса не менялись — тот же
`mdlm_sft/final`. Сэмплер v2 = confidence-based unmasking (MaskGIT) + repetition
penalty + neighbor-ban + no-repeat-ngram.

## Результат: repetition collapse = DECODE-проблема (решена)

| промпт | v1 sampler (random unmask) | v2 sampler (confidence + rep-control) |
|---|---|---|
| lonely | ❌ "people people people" | ✅ связный эмпатичный совет |
| panic attack | ❌ "attack attack attack" | ✅ реальные техники (дыхание, PMR, senses) |
| burnout | (ок) | ✅ связно, self-care/hobbies/breathing |
| anxious / procrastinating / sleep / emotionally-unavailable | (ок) | ✅ связно, структурированные советы |
| **CBT vs ACT** | ❌ "ACT-CT-CT" | ⚠️ seed0 связно но факты смазаны; seed1 всё ещё acronym-soup |
| **Vygotsky ZPD** | ❌ "proximalalks bulk" | ⚠️ коллапса нет, но ГАЛЛЮЦИНАЦИЯ: "coastal/coral/distal/shoreal" |

**7/9 coping-промптов:** collapse полностью ушёл, связность резко выросла. Это
доминирующий use-case (SFT-данные — в основном coping-диалоги).

→ **Repetition collapse был артефактом декода (random unmasking), не модели.**
Confidence-based unmasking + rep-penalty его устранил. По буквальному критерию
гейта — **decode был причиной, гейт открыт.**

## НО: вскрылся второй сигнал — контаминация корпуса (drift)

Остаточные проблемы на factual-промптах — это НЕ collapse, а **дрейф из данных**:
- **Vygotsky → "coastal/coral/distal/shoreal region"**: модель цепляет морскую
  биологию/экологию за слово "proximal/distal" (в анатомии/экологии те же термины).
- **CBT/ACT acronym-soup (seed1)**: частично факт-незнание, частично pile-шум.

Состав корпуса (подтверждено):
| источник | размер | доля |
|---|---|---|
| **pile_sample** (general text, psych-keyword filter) | **7.6 GB** | **~69%** |
| pes2o_sample (психо-статьи) | 1.6 GB | ~15% |
| openalex_psychology | 1.0 GB | ~9% |
| pmc/gutenberg/psyarxiv/translated | ~0.5 GB | ~7% |

**69% корпуса — шумно-отфильтрованный pile.** Vygotsky→coastal — прямая улика:
pile тащит off-domain (marine bio, ecology) который "proximal/distal" пускают в
психологическую семантику.

## Вывод и развилка для 120M

По **буквальному** правилу гейта (collapse ушёл → decode) — можно на текущем
корпусе. НО твоя глубинная цель была «не масштабировать дрейф», а дрейф мы
**нашли** — он data-driven (pile-контаминация), не decode.

- **Вариант A — 120M на текущем корпусе (2.56B):** быстро, collapse решён сэмплером,
  но 120M выучит и pile-дрейф (Vygotsky→coastal усилится).
- **Вариант B — почистить корпус, потом 120M:** ужесточить psych-фильтр pile
  (или дропнуть) → чище грунт, меньше галлюцинаций. Цена: rebuild ~2-3ч; токенов
  станет меньше (дроп всего pile → ~0.8B, мало для 120M; ужесточение → ~1.3-1.8B).

Токен-математика 120M (Chinchilla ~20 tok/param → ~2.4B оптимум):
- текущий 2.56B шумный = 21 tok/param, но с дрейфом
- чистый ~1.5B = 12 tok/param (чуть undertrained, но чистый сигнал)

## Рекомендация
Лёгкий перекос к **B (почистить)**: для психологической доменной модели factual
grounding важен, а Vygotsky→coastal — явный признак что pile вредит. Но A
защитимо — названный критерий гейта (collapse) закрыт. **Решение за тобой.**

## Артефакты
- `sample_mdlm_v2.py` — confidence-unmask + rep-control (рабочий, рекомендуемый дефолт)
- `mdlm_sft_samples_v2.txt` — 9 промптов × 2 seed
- 120M убит на step 220 (не на том корпусе пока не решено)

---
Date: 2026-05-29
Status: collapse=decode (решено). drift=data (pile 69%). 120M-корпус — развилка A/B.
