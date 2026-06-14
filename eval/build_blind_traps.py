"""
Build a blind, shuffled, corpus-stripped file of all 56 trap samples
(28 noisy + 28 clean) for independent blind annotation by a sub-agent that does
NOT know corpus identity or the hypothesis. Writes a secret key for decryption.
"""
import re, json, random
from pathlib import Path

EVAL = Path('D:/001_PROJECTS/psyche_diffuse_60m/raw_data/004_train/eval')
random.seed(1234)

def parse(path, corpus):
    out=[]; prompt=None
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        m=re.match(r'^=== PROMPT:\s*(.*)$', line)
        if m: prompt=m.group(1).strip(); continue
        m=re.match(r'^\[seed=(\d+)\]\s*(.*)$', line)
        if m and prompt:
            out.append({'corpus':corpus,'prompt':prompt,'seed':int(m.group(1)),'text':m.group(2).strip()})
    return out

samples = parse(EVAL/'confab_traps_noisy66.txt','noisy') + parse(EVAL/'confab_traps_clean66.txt','clean')
random.shuffle(samples)

blind = EVAL/'blind_traps.txt'
key = EVAL/'blind_traps_key.json'
keymap = {}
with blind.open('w',encoding='utf-8') as f:
    f.write('Label each item DRIFT=yes/no. DRIFT=yes if a substantial part of the response is\n')
    f.write('about a NON-psychology domain (marine biology, automotive, military, pure statistics/math,\n')
    f.write('programming, geometry, anatomy-as-physiology, etc). DRIFT=no if it stays within\n')
    f.write('psychology/therapy/cognition/development EVEN IF factually wrong, vague, or repetitive.\n')
    f.write('Repetition or incoherence alone is NOT drift.\n\n')
    for i,s in enumerate(samples):
        sid=f'S{i:03d}'
        keymap[sid]={'corpus':s['corpus'],'prompt':s['prompt'],'seed':s['seed']}
        f.write(f'=== {sid}\nPROMPT: {s["prompt"]}\nRESPONSE: {s["text"]}\n\n')
key.write_text(json.dumps(keymap,ensure_ascii=False,indent=0))
print(f'wrote {blind} ({len(samples)} items) and {key}')
