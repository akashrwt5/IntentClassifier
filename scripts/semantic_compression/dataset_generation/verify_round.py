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
# The set is known INCOMPLETE -- DEFERRED E3 counts 30 memory-name/intent overlaps,
# 17 unguarded. So this asserts a FLOOR, not an exact count. An exact count would
# fail the moment someone adds a guard E3 says is missing, which is the opposite of
# what a test should do.
add(f'15 F16 the seven original memory-name guards still present (found {len(coll)})',
    len(coll) >= 7 and {'Cmd.EdgeModeIncrease', 'Cmd.StreamingStart', 'Cmd.VolumeMute',
                        'Help_MaskMode', 'Help_Pairing', 'Help_Tinnitus'} <= set(coll))
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
add('23 Cmd.MemoryChange mirrors the memory-name rule (set incomplete, DEFERRED E3)',
    len(guards) >= 7 and 'Cmd.MemoryChange' in guards)

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

# --- HelpAppSettings family round ----------------------------------------
home, health = by['Help_Home'], by['Help_Health']
add('43 D8 orientation catch-all trigger removed from Help_Home',
    not any('broad orientation question' in x for x in home['trigger_conditions'])
    and len(home['trigger_conditions']) == 4)
add('44 D8 both dependent boundary cases rewritten',
    not any('general orientation fallback' in x for x in home['boundary_cases'])
    and any('naming no screen and no feature is Default Fallback' in x for x in home['boundary_cases']))
add('45 D8 quick start / overview named on the Help_WhatsNew exclusion',
    any('quick start, overview or getting-started' in x for x in home['do_not_trigger']))
add('46 E1 CLOSED -- Health/Home guarded on both sides',
    any('Help_Health' in x for x in home['do_not_trigger'])
    and any('Help_Home' in x for x in health['do_not_trigger']))
add('47 E1 CLOSED -- and linked, so it cannot drift back',
    'Help_Health' in home['neighbor_intents'] and 'Help_Home' in health['neighbor_intents'])
add('48 D9 DeviceSettings <-> Customize now mutual neighbours',
    'Help_Customize' in by['Help_DeviceSettings']['neighbor_intents']
    and 'Help_DeviceSettings' in by['Help_Customize']['neighbor_intents'])
add('49 the other four HelpAppSettings specs untouched',
    len(by['Help_AppSettings']['trigger_conditions']) == 5
    and len(by['Help_WhatsNew']['trigger_conditions']) == 4
    and len(by['Help_DemoMode']['trigger_conditions']) == 5
    and len(by['Help_DeviceSettings']['trigger_conditions']) == 7)

# --- HelpHealth family round ---------------------------------------------
hr, hrr = by['Help_HeartRate'], by['Help_HeartRateRecovery']
add('50 D10 the current-heart-rate VALUE trigger is gone',
    not any('current heart rate' in x for x in hr['trigger_conditions']) and len(hr['trigger_conditions']) == 4)
add('51 D10 the business description says it does not report the value',
    'does not report the value' in hr['business_description'])
add('52 D10 the boundary case gives the explaining-IS-the-action reason',
    any('when the requested action IS explaining' in x for x in hr['boundary_cases'])
    and not any('No Cmd intent exists for reading heart rate' in x for x in hr['boundary_cases']))
add('53 D11 both interpretation triggers removed from Help_HeartRateRecovery',
    not any('normal value' in x or 'improve their heart rate' in x for x in hrr['trigger_conditions'])
    and len(hrr['trigger_conditions']) == 3)
add('54 D11 business description no longer promises what a good value looks like',
    'good value' not in hrr['business_description'])
add('55 D11 routed to Fallback by name, from the HRR side',
    any('Default Fallback Intent' in x for x in hrr['do_not_trigger']))
add('56 D11 D4 widened -- Fallback covers a clinical READING, not just a condition',
    any('clinical reading such as a heart rate' in x for x in by[FB]['trigger_conditions']))
add('56b D11 correction -- the over-broad wording is gone',
    not any('measured health value' in x for x in by[FB]['trigger_conditions']))
add('56c D11 correction -- app scores explicitly carved out of the Fallback rule',
    any('app score is not a clinical reading' in x for x in by[FB]['trigger_conditions']))
add('56d D11 correction -- Help_ThriveScore untouched and still owns improving a score',
    any('improve or increase a score' in x for x in by['Help_ThriveScore']['trigger_conditions'])
    and len(by['Help_ThriveScore']['do_not_trigger']) == 3)
add('57 three vague heart-rate cross-references now name the intent',
    any('which are Help_HeartRateRecovery' in x for x in by['Help_Health']['do_not_trigger'])
    and any('prefer Help_HeartRate' in x for x in by['Help_Health']['boundary_cases'])
    and any('Help_HeartRate and Help_HeartRateRecovery' in x for x in by['Help_ThriveScore']['do_not_trigger']))
add('58 the other three HelpHealth specs untouched',
    len(by['Help_Activity']['trigger_conditions']) == 3
    and len(by['Help_FallAlert']['trigger_conditions']) == 6
    and len(by['Help_ThriveScore']['trigger_conditions']) == 5)

# --- HelpDeviceCare family round -----------------------------------------
sc = by['Help_SelfCheck']
add('59 D12 SelfCheck carve-out present, worded for THIS case',
    any('no voice command runs it' in x for x in sc['boundary_cases']))
add('60 D12 the generation-rate instruction is in the spec',
    any('generated rate near zero is a defect' in x for x in sc['boundary_cases']))
add('61 D12 it does NOT copy the "assistant cannot" wording, which is false here',
    not any('assistant cannot' in x for x in sc['boundary_cases']))
add('62 D13 CleanCare <-> SelfCheck mutual',
    'Help_SelfCheck' in by['Help_CleanCare']['neighbor_intents']
    and 'Help_CleanCare' in sc['neighbor_intents'])
add('63 D13 WiCROS <-> Volume mutual, and Volume now names the balance control',
    'Help_Volume' in by['Help_WiCROS']['neighbor_intents']
    and 'Help_WiCROS' in by['Help_Volume']['neighbor_intents']
    and any('balance control' in x for x in by['Help_Volume']['do_not_trigger']))
add('64 the other four HelpDeviceCare specs untouched',
    len(by['Help_Battery']['trigger_conditions']) == 4
    and len(by['Help_InsertDevice']['trigger_conditions']) == 5
    and len(by['Help_Accessories']['trigger_conditions']) == 6
    and len(by['Help_WiCROS']['trigger_conditions']) == 6)

# --- HelpConnectivity family round ---------------------------------------
add('65 D14 Help_Pairing carve-out cites the Cmd.StreamingStart rule it mirrors',
    any('Cmd.StreamingStart already routes' in x for x in by['Help_Pairing']['boundary_cases']))
add('66 D14 all three carry a measured command-shaped rate for generation',
    any('29.5% command-shaped' in x for x in by['Help_Pairing']['boundary_cases'])
    and any('12.6% command-shaped' in x for x in by['Help_RemoteProgramming']['boundary_cases'])
    and any('8.3% command-shaped' in x for x in by['Help_HearShare']['boundary_cases']))
add('67 D14 RemoteProgramming and HearShare say the assistant does NOT act',
    any('does not submit an adjustment request' in x for x in by['Help_RemoteProgramming']['boundary_cases'])
    and any('does not accept or send an invitation' in x for x in by['Help_HearShare']['boundary_cases']))
add('68 D15 three prose-only pairs made mutual',
    all(b in by[a]['neighbor_intents'] and a in by[b]['neighbor_intents'] for a, b in
        (('Help_Pairing', 'Help_HearShare'), ('Help_HearShare', 'Help_Health'),
         ('Help_RemoteProgramming', 'Help_Customize'))))
add('69 no trigger_conditions changed in this family',
    len(by['Help_Pairing']['trigger_conditions']) == 6
    and len(by['Help_RemoteProgramming']['trigger_conditions']) == 7
    and len(by['Help_HearShare']['trigger_conditions']) == 5)

# --- audit fixes, 2026-08-27 ---------------------------------------------
# Every check below pins a defect found by auditing this review's own committed
# work. They exist because 76 passing checks coexisted with 15 real defects --
# the checks asserted what was DONE, not that it was TRUE.
add('70 AUDIT ranking claims match generator_config, which had it right all along',
    any('second highest of any Help intent' in x for x in by['Help_Pairing']['boundary_cases'])
    and any('third highest of any Help intent' in x for x in by['Help_SelfCheck']['boundary_cases'])
    and not any('third highest in the taxonomy' in x for x in by['Help_Pairing']['boundary_cases'])
    and not any('second only to' in x for x in by['Help_SelfCheck']['boundary_cases']))
add('71 AUDIT no spec claims deployed rows are "entirely" one shape',
    not any('entirely question-shaped' in x
            for s in specs for x in s['boundary_cases'] + s['do_not_trigger']))
add('72 AUDIT Help_Health no longer routes broad app questions to Help_Home',
    not any('broad questions about the app' in x for x in by['Help_Health']['do_not_trigger'])
    and any('naming no screen and no feature is Default Fallback' in x
            for x in by['Help_Home']['boundary_cases']))
add('73 AUDIT Fallback <-> Help_HeartRate guarded, the pair D11s fix created',
    any('Help_HeartRate' in x for x in by[FB]['do_not_trigger']))
# The check that would have caught the ranking errors. Every percentage a spec
# asserts about its own deployed speech is re-derived from train.csv here, so a
# stale number fails the run instead of reaching the generation prompt.
try:
    import csv as _csv, collections as _c, boundary_lint as _bl
    _rows = list(_csv.DictReader(open(_bl.DEPLOYED)))
    _byi = _c.defaultdict(list)
    for _r in _rows:
        _byi[_r['intent']].append(_r['text'])
    _bad = []
    for _s in specs:
        _txt = [_s['business_description']] + _s['trigger_conditions'] + _s['do_not_trigger'] + _s['boundary_cases']
        for _x in _txt:
            for _m in re.finditer(r'(\d+\.?\d*)%\s*command-shaped', _x):
                _rs = _byi.get(_s['name'], [])
                if not _rs:
                    _bad.append((_s['name'], 'no deployed rows')); continue
                _cc = _c.Counter(_bl.surface_form(_t)[0] for _t in _rs)
                _act = 100 * _cc['command-shaped'] / len(_rs)
                if abs(_act - float(_m.group(1))) >= 0.05:
                    _bad.append((_s['name'], f"claims {_m.group(1)}%, actual {_act:.1f}%"))
    add(f'74 AUDIT every command-shaped percentage in a spec re-derives exactly {_bad or ""}', not _bad)
except Exception as _e:
    add(f'74 AUDIT percentage re-derivation could not run ({type(_e).__name__})', False)

# --- the generated report ------------------------------------------------
md = open(f'{D}/SPEC_REVIEW.md').read()
a = md.split('### 2a')[1].split('### 2b')[0]
add('31 Section 2a empty -- Health/Home now guarded on both sides (DEFERRED E1 closed)',
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
