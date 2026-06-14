"""
Blind pool for the SCALE comparison: 28 clean-120M + 28 clean-66M trap samples,
shuffled and corpus-stripped, for independent blind annotation by fresh sub-agents
that do NOT know which scale produced each item or the hypothesis. Writes a secret key.
(Adapted from build_blind_traps.py, which compared noisy66 vs clean66.)
"""
import re, json, random
from pathlib import Path

EVAL = Path('D:/001_PROJECTS/psyche_diffuse_60m/raw_data/004_train/eval')
random.seed(2026)

def parse(path, corpus):
    out = []; prompt = None
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        m = re.match(r'^=== PROMPT:\s*(.*)$', line)
        if m:
            prompt = m.group(1).strip(); continue
        m = re.match(r'^\[seed=(\d+)\]\s*(.*)$', line)
        if m and prompt:
            out.append({'corpus': corpus, 'prompt': prompt, 'seed': int(m.group(1)), 'text': m.group(2).strip()})
    return out

samples = parse(EVAL / 'confab_traps_clean120.txt', '120m') + parse(EVAL / 'confab_traps_clean66.txt', '66m')
random.shuffle(samples)

blind = EVAL / 'blind_traps_scale.txt'
key = EVAL / 'blind_traps_scale_key.json'
keymap = {}
with blind.open('w', encoding='utf-8') as f:
    f.write('Label each item DRIFT=yes/no. DRIFT=yes if a substantial part of the response is\n')
    f.write('about a NON-psychology domain (physics/cosmology, marine biology, automotive, military,\n')
    f.write('pure statistics/math, programming, geometry, economics, anatomy-as-physiology, etc).\n')
    f.write('DRIFT=no if it stays within psychology/therapy/cognition/development EVEN IF factually\n')
    f.write('wrong, vague, or repetitive. Repetition or incoherence alone is NOT drift.\n\n')
    for i, s in enumerate(samples):
        sid = f'S{i:03d}'
        keymap[sid] = {'corpus': s['corpus'], 'prompt': s['prompt'], 'seed': s['seed']}
        f.write(f'=== {sid}\nPROMPT: {s["prompt"]}\nRESPONSE: {s["text"]}\n\n')
key.write_text(json.dumps(keymap, ensure_ascii=False, indent=0))
print(f'wrote {blind.name} ({len(samples)} items: '
      f'{sum(1 for s in samples if s["corpus"]=="120m")} x120m + '
      f'{sum(1 for s in samples if s["corpus"]=="66m")} x66m) and {key.name}')
