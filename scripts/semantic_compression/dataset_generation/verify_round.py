"""Full verification, run against the REAL repo. No API, no network, reads specs only."""
import re, sys, yaml
from pathlib import Path

# Lives beside the specs it checks, so it runs from anywhere.
D = str(Path(__file__).resolve().parent)
ROOT = str(Path(D).parents[2])
sys.path.insert(0, D)
import bootstrap_specs as bs
import spec_review as sr  # noqa: F401  (import proves it loads)

I = yaml.safe_load(open(f'{D}/intent_specs.yaml')); specs = I['intents']
by = {s['name']: s for s in specs}
A = yaml.safe_load(open(f'{D}/authored_specs.yaml')); abn = {s['name']: s for s in A['intents']}
names = set(by); FB = 'Default Fallback Intent'
F = ['business_description', 'trigger_conditions', 'do_not_trigger', 'boundary_cases',
     'neighbor_intents', 'positive_example', 'hard_negative_example']
R = []; add = lambda l, v: R.append((l, bool(v)))

add('1  60 intents, unique names', len(specs) == 60 and len(names) == 60)
add('2  authored/intent_specs identical on 7 fields',
    set(abn) == names and all(abn[n].get(f) == by[n].get(f) for n in names for f in F))
add('3  authored_by present on all 60', all(abn[n].get('authored_by') for n in names))
add('4  129 comments preserved in authored_specs',
    sum(1 for l in open(f'{D}/authored_specs.yaml').read().splitlines() if l.strip().startswith('#')) == 129)
blob = ' '.join(x for n in names for f in F for x in ([by[n][f]] if isinstance(by[n][f], str) else by[n][f]))
add('5  no hyphen-fold corruption', not re.search(r'[a-z]- [a-z]', blob))
add('6  no unknown neighbour names', not [(n, x) for n, s in by.items() for x in s['neighbor_intents'] if x not in names])
add('7  no one-directional links (Fallback exempt)',
    not [(n, x) for n, s in by.items() for x in s['neighbor_intents'] if n not in by[x]['neighbor_intents'] and x != FB])
add('8  min 2 neighbours each', all(len(s['neighbor_intents']) >= 2 for s in specs))
add('9  no spec lists itself', not [n for n, s in by.items() if n in s['neighbor_intents']])

cfg = yaml.safe_load(open(f'{D}/generator_config.yaml'))
def fp(o):
    if isinstance(o, dict):
        if 'command_help_pairs' in o: return o['command_help_pairs']
        for v in o.values():
            r = fp(v)
            if r is not None: return r
pr = fp(cfg); pl = list(pr.items()) if isinstance(pr, dict) else [tuple(p) for p in pr]
add(f'10 {len(pl)}/{len(pl)} Cmd/Help pairs mutual neighbours',
    not [(c, h) for c, h in pl if h not in by[c]['neighbor_intents'] or c not in by[h]['neighbor_intents']])
add('11 Fallback is a neighbour of 59/59', sum(1 for n, s in by.items() if n != FB and FB in s['neighbor_intents']) == 59)

mf = set(bs.IntentSpecification.model_fields); ok = 0; probs = 0
for s in specs:
    o = bs.IntentSpecification(**{k: v for k, v in s.items() if k in mf}); ok += 1
    probs += len(bs.validate_spec(o, s['name'], names))
add('12 60/60 IntentSpecification + validate_spec clean', ok == 60 and probs == 0)

# --- Help_Tinnitus round -------------------------------------------------
t = by['Help_Tinnitus']
add('13 F17 clinical trigger removed', not any('ringing in their ears is' in x for x in t['trigger_conditions']))
add('14 F17 stated on BOTH sides',
    any('clinical' in x for x in t['do_not_trigger']) and any('medical or clinical' in x for x in by[FB]['trigger_conditions']))
coll = [n for n in names if any('both a memory name and' in x for x in by[n]['do_not_trigger'])]
add(f'15 F16 all 7 memory-name collisions guarded (found {len(coll)})', len(coll) == 7)
add('16 F16 three new mutual neighbour pairs',
    all(x in by['Cmd.MemoryChange']['neighbor_intents'] and 'Cmd.MemoryChange' in by[x]['neighbor_intents']
        for x in ('Help_Tinnitus', 'Help_MaskMode', 'Help_Pairing')))
add('17 Cmd.MemoryChange has 10 neighbours', len(by['Cmd.MemoryChange']['neighbor_intents']) == 10)

# --- Help_IntelliVoice round ---------------------------------------------
iv = by['Help_IntelliVoice']
add('18 F22 trigger widened to "when, or how often"', any('when, or how often' in x for x in iv['trigger_conditions']))
add('19 F22 old occasion-only trigger gone',
    not any(x == 'User asks when they should use IntelliVoice.' for x in iv['trigger_conditions']))
add('20 F18 precautionary boundary case present', any('no Cmd intent for IntelliVoice' in x for x in iv['boundary_cases']))
add('21 F20 capability trigger left intact', any('support IntelliVoice' in x for x in iv['trigger_conditions']))
add('22 Cmd.EdgeModeIncrease STILL UNCHANGED (6 do_not_trigger, 8 neighbours)',
    len(by['Cmd.EdgeModeIncrease']['do_not_trigger']) == 6 and len(by['Cmd.EdgeModeIncrease']['neighbor_intents']) == 8)
guards = {n for n, s in by.items() if any('both a memory name and' in x for x in s['do_not_trigger'])}
add('23 6 owning intents + Cmd.MemoryChange carry the memory-name rule',
    len(guards) == 7 and 'Cmd.MemoryChange' in guards)

# --- Help_MaskMode round -------------------------------------------------
mm = by['Help_MaskMode']
add('24 F23 face-mask rationale gone from business_description',
    'face mask' not in mm['business_description'].lower() and 'memory list' in mm['business_description'])
add('25 F23 face-mask-problems trigger removed',
    not any('face-mask-related' in x for x in mm['trigger_conditions']) and len(mm['trigger_conditions']) == 3)
add('26 F23 the correction is recorded, not just deleted',
    any('NOT a face-mask remedy' in x for x in mm['boundary_cases']))
add('26b D6 the false "no Cmd intent for Mask Mode" premise is gone',
    not any('no Cmd intent for Mask Mode' in x for x in mm['boundary_cases']))
add('26c D6 mode-naming rule stated in Help_MaskMode, both fields',
    any('Naming a mode is not a request to change program' in x for x in mm['boundary_cases'])
    and any('stays here even with an on or off verb' in x for x in mm['do_not_trigger']))
add('26d D6 mode-naming rule stated on the Cmd.MemoryChange side too',
    any('Naming a MODE is not a request to change program' in x
        for x in by['Cmd.MemoryChange']['boundary_cases']))
add('27 F24 Tinnitus <-> MaskMode mutual neighbours',
    'Help_Tinnitus' in mm['neighbor_intents'] and 'Help_MaskMode' in by['Help_Tinnitus']['neighbor_intents'])
add('28 F24 stated on both sides in do_not_trigger',
    any('tinnitus masker' in x for x in mm['do_not_trigger'])
    and any('Help_MaskMode' in x for x in by['Help_Tinnitus']['do_not_trigger']))
add('29 Mask is genuinely a runtime memory name (entity list)',
    'Mask' in __import__('json').load(open('language_packs/en/nlu_entities.json'))['memory']['values'])
add('30 no plain-scalar ": " introduced into authored_specs',
    all(': ' not in l.split('- ', 1)[1] for l in open(f'{D}/authored_specs.yaml').read().splitlines()
        if l.lstrip().startswith('- ') and l.startswith('      - ')))

# --- Default Fallback Intent round ---------------------------------------
fb = by[FB]
add('36 F27 Cmd.MemoryChange is now a Fallback neighbour', 'Cmd.MemoryChange' in fb['neighbor_intents'])
add('37 F27 stayed one-sided -- Cmd.MemoryChange untouched (10 neighbours)',
    len(by['Cmd.MemoryChange']['neighbor_intents']) == 10)
add('38 F27 the three real neighbours kept',
    all(x in fb['neighbor_intents'] for x in ('Cmd.StreamingStart', 'Help_Volume', 'reminders.add')))
add('39 reminders exclusion present on the Fallback side',
    any('reminders.add' in x for x in fb['do_not_trigger']))
add('40 WITHDRAWN FINDING -- the Section 6 reference is left intact',
    any('Section 6' in x for x in fb['trigger_conditions']))
add('41 Section 6 really is the blueprint precedence section',
    '## 6. Structured Ambiguity' in open(f'{ROOT}/docs/Prod-Work-Documentation/nlu_super_dataset_architecture.md').read())
add('42 reminders.add itself unchanged (still unreviewed)',
    len(by['reminders.add']['boundary_cases']) == 3 and len(by['reminders.add']['do_not_trigger']) == 4)

# --- the generated report ------------------------------------------------
md = open(f'{D}/SPEC_REVIEW.md').read()
a = md.split('### 2a')[1].split('### 2b')[0]
add('31 Section 2a empty -- Health/Home fell to 0.1999, NOT fixed (see DEFERRED E1)',
    'None' in a)
b = md.split('### 2b')[1].split('### 2c')[0]
# 2b is NOT empty and is not expected to be: the one pair below is logged in
# DEFERRED A1 as a queued edit against a deferred spec. Assert the exact known
# state so that a NEW pair appearing here fails the run.
add('32 Section 2b holds only the logged EdgeModeDeactivate/MaskMode pair',
    'Cmd.EdgeModeDeactivate' in b and 'Help_MaskMode' in b and '| 1 |' in b and '| 2 |' not in b)
add('32b Section 2c empty', 'both sides' in md.split('### 2c')[1].split('## 3')[0])
add('33 Section 3: 0 of 23 pairs failing', '**0 of 23 pairs are not mutual neighbours.**' in md)
add('34 sign-off boxes all unticked', md.count('☑') == 0 and md.count('☐') == 18)
add('35 REQUIRES HUMAN REVIEW still in meta', 'REQUIRES HUMAN REVIEW' in str(I.get('meta')))

for l, v in R: print(f'{l:<56}{"PASS" if v else "FAIL"}')
n = sum(1 for _, v in R if v)
print(f'\n{n}/{len(R)} pass | {sum(len(s["neighbor_intents"]) for s in specs)} neighbour links')
sys.exit(0 if n == len(R) else 1)
