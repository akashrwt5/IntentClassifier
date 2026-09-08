"""Full verification, run against the REAL repo. No API, no network, reads specs only."""
import ast as _ast
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

# 57 since D20 dropped the three unsupported intents. Was 60 for the review.
add('1  57 intents, unique names', len(specs) == 57 and len(names) == 57)
add('2  authored/intent_specs identical on 7 fields',
    set(abn) == names and all(abn[n].get(f) == by[n].get(f) for n in names for f in F))
add('3  authored_by present on all 57', all(abn[n].get('authored_by') for n in names))
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
add('11 Fallback is a neighbour of 56/56', sum(1 for n, s in by.items() if n != FB and FB in s['neighbor_intents']) == 56)

mf = set(bs.IntentSpecification.model_fields); ok = 0; probs = 0
for s in specs:
    o = bs.IntentSpecification(**{k: v for k, v in s.items() if k in mf}); ok += 1
    probs += len(bs.validate_spec(o, s['name'], names))
add('12 57/57 IntentSpecification + validate_spec clean', ok == 57 and probs == 0)

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
# 12 since D18 added the Cmd.TranscribeStart link, 11 after D17's
# Cmd.EdgeModeDeactivate, 10 for the F16/F27 rounds. Check 37 pins the names.
add('17 Cmd.MemoryChange has 12 neighbours', len(by['Cmd.MemoryChange']['neighbor_intents']) == 12)

# --- Help_IntelliVoice round ---------------------------------------------
iv = by['Help_IntelliVoice']
add('18 F22 trigger widened to "when, or how often"', any('when, or how often' in x for x in iv['trigger_conditions']))
add('19 F22 old occasion-only trigger gone',
    not any(x == 'User asks when they should use IntelliVoice.' for x in iv['trigger_conditions']))
add('20 F18 precautionary boundary case present', any('no Cmd intent for IntelliVoice' in x for x in iv['boundary_cases']))
add('21 F20 capability trigger left intact', any('support IntelliVoice' in x for x in iv['trigger_conditions']))
# Was '6 do_not_trigger, 8 neighbours -- STILL UNCHANGED' while EdgeMode was
# deferred. D17 read the family and added the seventh exclusion (queued A1 edit
# 1). The neighbour count is deliberately still 8: A1 edit 2 was declined, so
# the IntelliVoice and Mask Mode boundaries stay prose-only. Check 98 pins that.
add('22 Cmd.EdgeModeIncrease after D17 (7 do_not_trigger, 8 neighbours)',
    len(by['Cmd.EdgeModeIncrease']['do_not_trigger']) == 7 and len(by['Cmd.EdgeModeIncrease']['neighbor_intents']) == 8)
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
# Was 'Cmd.MemoryChange untouched (10 neighbours)'. D17 touched it, so the
# premise is retired and the list is pinned BY NAME instead -- Fallback appears
# in all 59, so its presence here proves nothing about F27. The eleventh entry
# is D17's, and any twelfth fails the run.
add('37 Cmd.MemoryChange neighbours are the F16 set plus the two memory-name rounds',
    set(by['Cmd.MemoryChange']['neighbor_intents']) == {
        'Help_ChangingMemories', 'Help_MemoryOptions', 'Cmd.EdgeModeIncrease',
        'Help_Customize', FB, 'Cmd.VolumeMute', 'Cmd.StreamingStart',
        'Help_Tinnitus', 'Help_MaskMode', 'Help_Pairing',
        'Cmd.EdgeModeDeactivate',   # D17, the bare back-to-normal tie
        'Cmd.TranscribeStart'})     # D18, the Meeting memory name
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

# --- HelpHealth family round, as it stands after D20 ---------------------
# Checks 50-55, 56c, 56d and 57 are RETIRED, not deleted quietly: they asserted
# D10 and D11 edits to Help_HeartRate, Help_HeartRateRecovery and
# Help_ThriveScore, and D20 dropped all three. What survived them is the
# Fallback widening those rounds produced, which outlives the intents.
add('56 D11 D4 widened -- Fallback covers a clinical READING, not just a condition',
    any('clinical reading such as a heart rate' in x for x in by[FB]['trigger_conditions']))
add('56b D11 correction -- the over-broad wording is gone',
    not any('measured health value' in x for x in by[FB]['trigger_conditions']))
# Was 'the surviving HelpHealth specs untouched', which was false the moment it
# was written: D20 edited Help_Health and Help_Activity. Now pins the shape D20
# left them in, so a later round cannot quietly widen either.
add('58 the surviving HelpHealth specs are the shape D20 left them in',
    len(by['Help_Activity']['trigger_conditions']) == 3
    and len(by['Help_Activity']['do_not_trigger']) == 3
    and len(by['Help_FallAlert']['trigger_conditions']) == 6
    and len(by['Help_Health']['trigger_conditions']) == 4
    and len(by['Help_Health']['do_not_trigger']) == 5
    and len(by['Help_Health']['boundary_cases']) == 3)

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
# 73 retired by D20. It asserted Fallback disclaimed Help_HeartRate, which no
# longer exists -- Fallback now CLAIMS the subject outright. Check 111 covers it.
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
            # Same guard for the other direction. D18 put a question-shaped rate on
            # a Cmd intent for the first time, so it needs re-deriving too.
            # question-shaped = help-shaped + explain-request, per boundary_lint.
            for _m in re.finditer(r'(\d+\.?\d*)%\s*question-shaped', _x):
                _rs = _byi.get(_s['name'], [])
                if not _rs:
                    _bad.append((_s['name'], 'no deployed rows')); continue
                _cc = _c.Counter(_bl.surface_form(_t)[0] for _t in _rs)
                _act = 100 * (_cc['help-shaped'] + _cc['explain-request']) / len(_rs)
                if abs(_act - float(_m.group(1))) >= 0.05:
                    _bad.append((_s['name'], f"claims {_m.group(1)}% question, actual {_act:.1f}%"))
    # A spec may also cite ANOTHER intent's rate -- "Cmd.TranscribeStart at
    # 2.0%", "Help_FindMyHearingAids at 40.6%". Nothing re-derived those, which
    # is how a false comparative reached a spec twice: once in the audit round
    # and once in D18. Accept either shape measure, since the sentence decides
    # which is meant and this only needs to catch a fabricated number.
    for _s in specs:
        _txt = ([_s['business_description']] + _s['trigger_conditions']
                + _s['do_not_trigger'] + _s['boundary_cases'])
        for _x in _txt:
            for _m in re.finditer(r'((?:Cmd|Help)[._]\w+) at (\d+\.?\d*)%', _x):
                _rs = _byi.get(_m.group(1), [])
                if not _rs:
                    _bad.append((_s['name'], f'cites {_m.group(1)}, which has no deployed rows'))
                    continue
                _cc = _c.Counter(_bl.surface_form(_t)[0] for _t in _rs)
                _cand = (100 * _cc['command-shaped'] / len(_rs),
                         100 * (_cc['help-shaped'] + _cc['explain-request']) / len(_rs))
                if not any(abs(_v - float(_m.group(2))) < 0.05 for _v in _cand):
                    _bad.append((_s['name'], f'cites {_m.group(1)} at {_m.group(2)}%, '
                                             f'actual {_cand[0]:.1f}/{_cand[1]:.1f}'))
    add(f'74 AUDIT every percentage a spec states or cites re-derives exactly {_bad or ""}', not _bad)
except Exception as _e:
    add(f'74 AUDIT percentage re-derivation could not run ({type(_e).__name__})', False)

# --- E8, defects found inside families already signed off ----------------
add('75 E8-1 all Cmd.Activity* now route locating to Help_Activity, not Help_Health',
    not any('Help_Health' in x for n, s in by.items() if n.startswith('Cmd.Activity')
            for f in ('do_not_trigger', 'boundary_cases') for x in s[f])
    and sum(1 for n, s in by.items() if n.startswith('Cmd.Activity')
            for f in ('do_not_trigger', 'boundary_cases') for x in s[f] if 'Help_Activity' in x) >= 16)
add('76 E8-1 Help_Activity claims it and no longer defers to Help_Health',
    any("tracked activity's data is shown" in x for x in by['Help_Activity']['trigger_conditions'])
    and not any('prefer Help_Health' in x for x in by['Help_Activity']['boundary_cases']))
add('77 E8-1 Help_Health yields it back by name',
    any('which are Help_Activity' in x for x in by['Help_Health']['do_not_trigger']))
add('78 E8-2 Help_SelfCheck no longer claims a faint aid against its own rule',
    not any('faint' in x for x in by['Help_SelfCheck']['trigger_conditions'])
    and any('too quiet is a volume request' in x for x in by['Help_SelfCheck']['boundary_cases']))
add('79 E8-3 Help_Volume <-> Help_Pairing guarded on both sides and linked',
    any('Help_Pairing' in x for x in by['Help_Volume']['do_not_trigger'])
    and any('Help_Volume' in x for x in by['Help_Pairing']['do_not_trigger'])
    and 'Help_Pairing' in by['Help_Volume']['neighbor_intents'])
add('80 E8 no Cmd.Activity* lost its Help_Activity neighbour link',
    all('Help_Activity' in by[n]['neighbor_intents'] for n in by if n.startswith('Cmd.Activity')))

# --- E9, config against specs --------------------------------------------
_cfg = yaml.safe_load(open(f'{D}/generator_config.yaml'))
def _find(o, k):
    if isinstance(o, dict):
        for kk, v in o.items():
            if kk == k: return v
            r = _find(v, k)
            if r is not None: return r
_pairs = _find(_cfg, 'command_help_pairs')
add('81 E9 the false messaging pairs are gone from command_help_pairs',
    not [k for k in _pairs if 'Message' in k] and len(_pairs) == 21)
add('82 E9 but the messaging confusion keeps its sampling -- family and neighbours',
    'Help_VoiceAssistant' in _find(_cfg, 'families')['Messaging']
    and 'Help_VoiceAssistant' in by['Cmd.SendMessage']['neighbor_intents']
    and 'Help_VoiceAssistant' in by['Cmd.ListenMessage']['neighbor_intents'])
add('83 E9 config no longer contradicts what the three specs say',
    all(any('no messaging Help intent' in x or 'has no messaging Help intent' in x
            for x in by[n]['do_not_trigger'] + by[n]['boundary_cases'])
        for n in ('Cmd.SendMessage', 'Cmd.ListenMessage')))
add('84 E9 provenance counts match the 57 specs that exist',
    len(_find(_cfg, 'hand_authored_intents')) == 57
    and 'All 57 are currently listed' in open(f'{D}/generator_config.yaml').read()
    and 'The remaining 56 were drafted' in open(f'{D}/authored_specs.yaml').read())

# --- E10, the two checks the audit said were missing ---------------------
# Sections 7 and 8 of spec_review.py. Both baselines are pinned by NAME, not by
# count, so a NEW route or collision fails the run while the known ones do not
# nag. Anything listed here is next round's work, recorded in DEFERRED E10.
_ow = {(r['from'], r['to']) for r in sr.one_way_routes(specs)}
_co = {(r['a'], r['b']) for r in sr.subject_collisions(specs)}
_OW_KNOWN = {('Help_AppSettings', 'Help_WhatsNew'),
             ('Help_HearShare', 'Help_RemoteProgramming'),
             ('Help_IntelliVoice', 'Help_MaskMode'),
             ('Help_RemoteProgramming', 'Help_Customize')}
# ('Cmd.EdgeModeDeactivate', 'Cmd.MemoryChange') removed by D17: the two now
# name each other over the bare back-to-normal tie. 7 -> 6.
_CO_KNOWN = {('Cmd.ListenMessage', 'reminders.complete'),
             ('Cmd.VolumeMute', 'Help_Tinnitus'),
             # D20 removed two: (FB, 'Help_ThriveScore') went with the intent,
             # and (FB, 'Help_Health') closed because Help_Health's rewritten
             # exclusion now names Default Fallback Intent outright. 6 -> 4.
             (FB, 'Help_HearShare'),
             ('Help_MemoryOptions', 'reminders.add')}
add(f'85 E10 Section 7 finds no NEW route without a destination {sorted(_ow - _OW_KNOWN)}',
    not (_ow - _OW_KNOWN))
add(f'86 E10 Section 8 finds no NEW unguarded collision {sorted(_co - _CO_KNOWN)}',
    not (_co - _CO_KNOWN))
add(f'87 E10 and the known ones have not been quietly fixed without updating this list '
    f'{sorted((_OW_KNOWN - _ow) | (_CO_KNOWN - _co))}',
    not ((_OW_KNOWN - _ow) or (_CO_KNOWN - _co)))

# --- D16, the last seven intents -----------------------------------------
add('88 D16 Help_MemoryOptions carve-out, with the reason and the measured rate',
    any('No command creates a memory' in x for x in by['Help_MemoryOptions']['boundary_cases'])
    and any('18.1% command-shaped' in x for x in by['Help_MemoryOptions']['boundary_cases']))
add('89 D16 the memory-availability split is stated on both sides',
    any('MISSING from the list' in x for x in by['Help_ChangingMemories']['do_not_trigger'])
    and any('whether a particular memory is available' in x for x in by['Help_MemoryOptions']['do_not_trigger']))
add('90 D16 Help_VoiceAssistant names Transcribe and Translate instead of "their own intents"',
    any('Help_Transcribe' in x and 'Help_Translate' in x for x in by['Help_VoiceAssistant']['do_not_trigger'])
    and not any('which have their own intents' in x for x in by['Help_VoiceAssistant']['do_not_trigger']))
add('91 D16 the reminders.complete one-way route is closed on the Fallback side',
    any('delete, cancel or postpone a reminder' in x for x in by[FB]['trigger_conditions']))
add('92 D16 two prose-only links made mutual',
    'Help_Pairing' in by['Help_FindMyHearingAids']['neighbor_intents']
    and 'Help_FindMyHearingAids' in by['Help_Pairing']['neighbor_intents']
    and 'Help_Customize' in by['Help_ChangingMemories']['neighbor_intents']
    and 'Help_ChangingMemories' in by['Help_Customize']['neighbor_intents'])
add('93 D16 Help_FindMyHearingAids left alone -- its carve-out was already the model',
    len(by['Help_FindMyHearingAids']['trigger_conditions']) == 4
    and any('40.6% command-shaped' in x for x in by['Help_FindMyHearingAids']['boundary_cases']))

# --- D17, the EdgeMode family -------------------------------------------
ei, ed, hem = by['Cmd.EdgeModeIncrease'], by['Cmd.EdgeModeDeactivate'], by['Help_EdgeMode']
cm2, mm2 = by['Cmd.MemoryChange'], by['Help_MaskMode']

add('94 D17 EdgeModeDeactivate routes a memory switch to the COMMAND, not the Help intent',
    any('which are Cmd.MemoryChange' in x for x in ed['do_not_trigger'])
    and not any('program, which are Help_ChangingMemories' in x for x in ed['do_not_trigger']))

# Akash: bare "back to normal" names two intents that can both perform it, so
# the tie goes to inaction. All three specs must say it, or generation will
# write the sentence under whichever label it happens to be generating.
add('95 D17 the bare back-to-normal tie goes to Fallback, stated in all three specs',
    any('bare return to normal' in x for x in ed['boundary_cases'])
    and any('bare return to normal' in x for x in cm2['boundary_cases'])
    and any('back to normal' in x for x in fb['trigger_conditions'])
    and any('Cmd.EdgeModeDeactivate' in x for x in cm2['boundary_cases'])
    and any('Cmd.MemoryChange' in x for x in ed['boundary_cases']))
add('95b D17 Cmd.MemoryChange no longer claims a bare "normal"',
    not any(x == 'User asks to return to normal, default or automatic.'
            for x in cm2['trigger_conditions']))
add('95c D17 and Cmd.EdgeModeDeactivate needs Edge Mode named for it',
    any('Edge Mode or adaptive tuning named' in x for x in ed['trigger_conditions']))
add('95d D17 the how-to side of it is left with Help_ChangingMemories',
    any('Help_ChangingMemories' in x for x in fb['do_not_trigger']))

# Akash: a bare environment observation stays Fallback. The exception needs a
# PLACE, because the memories are environment-scoped; a remark on how the sound
# is names no place.
add('96 D17 bare environment observation stays Fallback, stated in all three specs',
    any('named PLACE is the single exception' in x for x in ei['boundary_cases'])
    and any('exception needs a PLACE named' in x for x in fb['do_not_trigger'])
    and any('names no place and is Default Fallback Intent' in x for x in cm2['boundary_cases']))

add('97 D17 queued A1 edit 1 applied -- EdgeModeIncrease names IntelliVoice and Mask Mode',
    any('Help_IntelliVoice' in x and 'Help_MaskMode' in x for x in ei['do_not_trigger']))
# DELIBERATE. Akash declined A1 edit 2, so this boundary never reaches the
# prompt as a confusion and is never sampled for hard negatives. Asserted so
# that it is a recorded decision rather than an omission nobody noticed.
add('98 D17 queued A1 edit 2 deliberately NOT applied -- the pair stays prose-only',
    'Help_IntelliVoice' not in ei['neighbor_intents']
    and 'Help_MaskMode' not in ei['neighbor_intents']
    and 'Cmd.EdgeModeIncrease' not in by['Help_IntelliVoice']['neighbor_intents']
    and 'Cmd.EdgeModeIncrease' not in mm2['neighbor_intents'])
add('99 D17 queued A1 edit 3 applied -- named on both sides AND mutual',
    any('Help_MaskMode' in x for x in ed['do_not_trigger'])
    and any('Cmd.EdgeModeDeactivate' in x for x in mm2['do_not_trigger'])
    and 'Help_MaskMode' in ed['neighbor_intents']
    and 'Cmd.EdgeModeDeactivate' in mm2['neighbor_intents'])

# 20 of Cmd.EdgeModeIncrease's 152 seeds name comfort or communication, 8 of
# them with an increase word, while only Cmd.EdgeModeDecrease claimed the axis.
add('100 D17 the comfort/communication axis is mirrored on both direction intents',
    any('comfort or communication' in x for x in ei['trigger_conditions'])
    and any('comfort or communication' in x for x in by['Cmd.EdgeModeDecrease']['trigger_conditions']))
# Help_EdgeMode's 19 seeds are 100% help-shaped, so unlike the five Help intents
# that got a command carve-out this one needs none. Assert it was left alone.
add('101 D17 Help_EdgeMode untouched -- no carve-out, none warranted',
    len(hem['trigger_conditions']) == 4 and len(hem['do_not_trigger']) == 3
    and len(hem['boundary_cases']) == 3 and len(hem['neighbor_intents']) == 7)

# --- D18, the SpeechServices family --------------------------------------
_tr, _ts = by['Cmd.TranscribeStart'], by['Cmd.TranslationStart']
_htr, _hts = by['Help_Transcribe'], by['Help_Translate']

# First question-shaped carve-out in the review, and the first on a Cmd intent.
# The rate itself is re-derived by check 74; this only asserts it is stated and
# that generation is told to reproduce it.
add('102 D18 Cmd.TranslationStart carries a measured question-shaped rate for generation',
    any('17.5% question-shaped' in x and 'near zero is a defect' in x
        for x in _ts['boundary_cases']))
# E3 listed Meeting -> Cmd.TranscribeStart among the overlaps that "look real".
# 48 of Cmd.MemoryChange's 1,601 deployed rows name a meeting against 5 of this
# intent's 50, so the memory reading is the common one and needs the guard.
add('103 D18 the Meeting memory-name guard, worded like the other six',
    any('both a memory name and' in x and 'Meeting' in x for x in _tr['do_not_trigger'])
    and any('Cmd.TranscribeStart' in x for x in by['Cmd.MemoryChange']['do_not_trigger'])
    and 'Cmd.MemoryChange' in _tr['neighbor_intents']
    and 'Cmd.TranscribeStart' in by['Cmd.MemoryChange']['neighbor_intents'])
# Cmd.SendMessage and Cmd.ListenMessage both named this intent; it named neither,
# while 11 of its 50 deployed rows carry "record" against 7 of SendMessage's 154.
add('104 D18 the record-to-transcribe vs record-to-send boundary is now stated from this side',
    any('Cmd.SendMessage' in x for x in _tr['do_not_trigger']))
# A2's one known dependency, checked from the destination: Cmd.ListenMessage
# sends live transcription here and this intent's triggers do claim it.
add('105 D18 A2 dependency holds -- Cmd.TranscribeStart claims what ListenMessage sends it',
    any('transcrib' in x.lower() for x in _tr['trigger_conditions'])
    and any('conversation' in x.lower() for x in _tr['trigger_conditions'])
    and any('Cmd.TranscribeStart' in x for x in by['Cmd.ListenMessage']['do_not_trigger']))
# Both Help specs are 0.0% command-shaped on 70 and 66 deployed rows, so unlike
# the five Help intents that needed a carve-out these two need nothing. Assert
# they were left alone rather than "improved" for symmetry.
add('106 D18 Help_Transcribe and Help_Translate untouched -- no carve-out warranted',
    len(_htr['trigger_conditions']) == 5 and len(_htr['do_not_trigger']) == 3
    and len(_htr['boundary_cases']) == 3
    and len(_hts['trigger_conditions']) == 6 and len(_hts['do_not_trigger']) == 2
    and len(_hts['boundary_cases']) == 2)
# Akash: transcription needs no stop action, so the "no stop intent exists"
# route to Fallback is left generic by decision, not by oversight. See D18.
add('107 D18 the stop-transcribing route is left as it was, by decision',
    any('No stop intent exists in this taxonomy' in x for x in _tr['do_not_trigger']))
# Section 7 was blind to the bare "Fallback" spelling. The matcher now knows it,
# and Cmd.FindMyPhone -- the one real route that surfaced -- is closed.
add('108 D18 spec_review knows the bare "Fallback" spelling',
    sr.INTENT_IN_TEXT.findall('which map to Fallback.') == ['Fallback'])
add('109 D18 the route it surfaced is closed on the Fallback side',
    any('other than the phone or the hearing aids' in x for x in fb['trigger_conditions'])
    and any('Default Fallback Intent' in x for x in by['Cmd.FindMyPhone']['do_not_trigger']))

# --- D20, dropping the three unsupported intents -------------------------
_D3 = ('Help_HeartRate', 'Help_HeartRateRecovery', 'Help_ThriveScore')
_cfgtxt = open(f'{D}/generator_config.yaml').read()
add('110 D20 all three gone from the taxonomy itself',
    not [n for n in _D3 if n in names]
    and not [n for n in _D3 if n in _find(_cfg, 'hand_authored_intents')]
    and len(_find(_cfg, 'families')['HelpHealth']) == 3)
# Fallback must CLAIM them, not merely be where they land by absence. Same
# standard C1 and C2 set for messaging and powering the aids on or off.
add('111 D20 Fallback states the unsupported set explicitly',
    any(all(w in x for w in ('heart rate', 'Thrive', 'None of the three is a supported'))
        for x in by[FB]['trigger_conditions'])
    and not any('which are Help_HeartRate' in x for x in by[FB]['do_not_trigger']))
# That carve-out existed only to keep app scores with Help_ThriveScore. With the
# intent gone it would send scores nowhere, so it had to go in the same change.
add('112 D20 the app-score carve-out went with the intent it protected',
    not any('app score is not a clinical reading' in x for x in by[FB]['trigger_conditions']))
add('113 D20 Help_Health and Help_Activity route them to Fallback and lost the links',
    any('None of the three is a supported feature' in x
        for x in by['Help_Health']['do_not_trigger'])
    and any('none of the three is a supported feature' in x
            for x in by['Help_Activity']['do_not_trigger'])
    and not set(by['Help_Health']['neighbor_intents']) & set(_D3)
    and not set(by['Help_Activity']['neighbor_intents']) & set(_D3))
# Dropping an intent that is still in the shipping label map is not the same as
# dropping one that was already gone. The config has to say so.
add('114 D20 drop_intents records all three, with the still-in-runtime warning',
    all(n in _find(_cfg, 'drop_intents') for n in _D3)
    and all(_find(_cfg, 'drop_intents')[n].get('reason') for n in _D3)
    and 'STILL in the shipping label map'
        in _find(_cfg, 'drop_intents')['Help_HeartRate'].get('note', '')
    and all('nlu_schema.json' in _find(_cfg, 'drop_intents')[n].get('note', '')
            for n in _D3))
add('115 D20 no spec anywhere still names one of them',
    not [(n, f) for n, s_ in by.items() for f in
         ('business_description', 'trigger_conditions', 'do_not_trigger', 'boundary_cases')
         for x in ([s_[f]] if isinstance(s_[f], str) else s_[f])
         for d3 in _D3 if d3 in x])
add('116 D20 length_targets carries no entry for them',
    not [n for n in _D3
         if n in yaml.safe_load(open(f'{D}/length_targets.yaml'))['intents']])

# --- D21, the defects this session's own review found --------------------
# A spec asserted "the highest of any action command in the taxonomy" for a
# 17.5% question-shaped rate. Cmd.FindMyPhone is 20.2% and is an action command
# -- it rings the phone. The claim came from an analysis that excluded it via a
# hardcoded status-query list the spec never disclosed. Superlatives about
# deployed speech are the defect class check 70 exists for, and this is the
# third one this review has produced.
add('117 D21 no spec claims a taxonomy-wide superlative about its own rate',
    not [s['name'] for s in specs for x in s['boundary_cases']
         if 'highest of any action command' in x or 'highest in the taxonomy' in x])
add('118 D21 the surviving comparison names the sibling and disclaims the ranking',
    any('the sibling it shares a family with' in x and 'not a taxonomy record' in x
        for x in by['Cmd.TranslationStart']['boundary_cases']))
# A dangling "- but" mid-sentence, left by the audit round. Check 5's corruption
# regex looks for "[a-z]- [a-z]" and cannot see ". - but".
add('119 D21 no spec has a list marker stranded mid-sentence',
    not [s['name'] for s in specs for f in ('trigger_conditions', 'do_not_trigger', 'boundary_cases')
         for x in s[f] if re.search(r'[.;]\s+-\s+[a-z]', x)])

# --- E11, the Fallback privacy substitute block --------------------------
# Nothing verified this block against the specs, and D20 put them in direct
# contradiction: the SCOPE GUARD listed "activity/health queries" as supported
# and said NEVER to generate a request for a supported capability, while D20
# had just given Fallback a trigger claiming heart rate, HRR and Thrive. The
# block steering Fallback generation forbade the utterances Fallback now owns.
#
# It is hand-written on purpose -- Fallback's 613 seeds are raw production ASR
# and are withheld under generation.privacy.no_seed_block_intents, so this text
# is the ONLY thing telling the model how the intent sounds. Nothing in it is a
# transcript or a paraphrase of one.
_priv = ((_cfg.get('generation') or {}).get('privacy') or {})
_sub = (_priv.get('seed_block_replacement') or {}).get(FB, '')
add('120 E11 Fallback is still the one intent whose seed block is withheld',
    _priv.get('no_seed_block_intents') == [FB] and len(_sub) > 500)
add('121 E11 the scope guard no longer calls health queries supported',
    'activity/health queries' not in _sub
    and 'activity tracking' in _sub
    and 'Health screen and\nits goals' in _sub)
add('122 E11 and it names the three unsupported subjects as belonging here',
    'THE EXCEPTION to the scope guard' in _sub
    and all(w in _sub for w in ('Heart rate', 'heart rate recovery', 'Thrive')))
# The contradiction in general form: a subject Fallback's own triggers claim
# must not sit in the guard's supported-and-never-generate list.
_supported = _sub.split('product ITSELF supports:')[-1].split('NEVER generate')[0].lower()
add('123 E11 no subject Fallback claims is listed as a supported capability',
    not [w for w in ('heart rate', 'thrive', 'wellness score')
         if w in ' '.join(by[FB]['trigger_conditions']).lower() and w in _supported])
add('124 E11 the dead redistribute_seeds_to key is gone from every drop entry',
    not [n for n, v in _find(_cfg, 'drop_intents').items() if 'redistribute_seeds_to' in v]
    and 'redistribute_seeds_to' in open(f'{D}/generator_config.yaml').read())

# --- D23, the quota that overruled five specs ----------------------------
# The 2026-08-30 pilot generated 0 of 26 and 0 of 25 command-shaped rows for
# the two highest carve-outs. Not model disobedience: the prompt carried the
# carve-out prose AND "at least 22 must have type `Question`" from the help
# profile's 0.88 floor, which leaves at most 3 rows of 25 for every other type.
# Four of the five carve-out rates were arithmetically impossible.
_gen = _cfg.get('generation') or {}
_q = _gen.get('quotas') or {}
_per = _q.get('per_intent') or {}
# Detect on the STATED RATE, exactly as boundary_lint does, and assert the two
# agree. The first version matched the phrase 'near zero is a defect' and missed
# Help_HearShare and Help_RemoteProgramming, whose carve-outs are worded
# differently -- a mutation test found it, and finding it added a sixth intent
# to the fix.
import boundary_lint as _bl2
_RATE = re.compile(r'\d+\.?\d*% command-shaped')
_carve = {s_['name']: s_ for s_ in specs
          if s_['name'].startswith('Help')
          and any(_RATE.search(x) for x in s_['boundary_cases'])}
add(f'124b D23 verify_round and boundary_lint agree on which intents are carve-outs '
    f'{sorted(set(_carve) ^ _bl2.carve_out_intents())}',
    set(_carve) == _bl2.carve_out_intents() and len(_carve) == 6)
add(f'125 D23 every carve-out spec has a per-intent quota override {sorted(set(_carve) - set(_per))}',
    len(_carve) >= 6 and not (set(_carve) - set(_per)))
# The override must ASK for the thing, not merely make room for it. This file's
# own lesson, stated twice in its comments: quota what the model under-produces.
# .get(n, {}) rather than [n]: a MISSING override is check 125's finding, and
# 126 crashing on it killed the whole run before 125 could print. A check must
# not take the report down with it.
add('126 D23 each Help override floors ExplicitCommand, not just lowers Question',
    all('ExplicitCommand' in ((_per.get(n) or {}).get('types') or {}) for n in _carve))
# The floor has to match the rate the spec states, which check 74 re-derives
# from train.csv -- so the config cannot drift away from the specs silently.
_bad_q = []
_BATCH = int(_gen.get('batch_size', 25))
for _n, _s in _carve.items():
    if not _n.startswith('Help'):
        continue
    _m = re.search(r'(\d+\.?\d*)% command-shaped', ' '.join(_s['boundary_cases']))
    if not _m:
        _bad_q.append((_n, 'spec states no rate')); continue
    _want = int(float(_m.group(1)) / 100 * _BATCH)
    _floor = ((_per.get(_n) or {}).get('types') or {}).get('ExplicitCommand')
    if not _floor:
        _bad_q.append((_n, 'no ExplicitCommand floor')); continue
    _got = int(_floor[0] * _BATCH)
    if _got != _want:
        _bad_q.append((_n, f'spec {_m.group(1)}% = {_want} rows, quota floors {_got}'))
add(f'127 D23 each ExplicitCommand floor equals the rate its own spec states {_bad_q}',
    not _bad_q)
# And the floors must fit. This is the arithmetic nobody did the first time.
add('128 D23 no override over-subscribes the batch',
    not [n for n in _per
         if sum(int(v[0] * _BATCH) for v in (_per[n].get('types') or {}).values()) > _BATCH])
# The lint that marked both failures "ok" is now two-sided for carve-outs only.
# Was a grep for 'def p_at_most', which 'def p_at_most_DISABLED' also satisfies
# -- a mutation test walked straight through it. Call the function instead.
_p_lo = getattr(_bl2, 'p_at_most', None)
add('129 D23 boundary_lint really tests the other tail, and the maths is right',
    callable(_p_lo)
    # 0 of 26 against a deployed 40.6% is the pilot's actual result. It must be
    # overwhelmingly improbable, or the test cannot fail what it was built for.
    and _p_lo(0, 26, 0.406) < 0.0001
    # and it must NOT fire on a rate that genuinely matches deployed speech.
    and _p_lo(10, 26, 0.406) > 0.05
    and 'FAIL (UNDER)' in open(f'{D}/boundary_lint.py').read())
# A subset pilot must never be able to write into the corpus directory. That is
# the one property the --pilot block exists to guarantee, and --only now runs
# inside it rather than around it.
_gsrc = open(f'{D}/generator.py').read()
add('129b D23 --only narrows the pilot instead of bypassing it',
    'must name intents IN the pilot set' in _gsrc
    and _gsrc.index('args.only = subset') < _gsrc.index('"checkpoint_dir"] = (')
    and 'args.force = True' in _gsrc)
add('130 D23 and it reads the carve-out list from the specs, not a hardcoded list',
    _bl2.carve_out_intents() and 'intent_specs.yaml' in open(f'{D}/boundary_lint.py').read())

# --- D24, the SHAPE of a direct request ----------------------------------
# The quota fix moved the type labels 0 -> 13 and 0 -> 11, and boundary_lint
# still failed both: the model labelled rows ExplicitCommand while writing them
# as "show me how to ..." and "I can't find them". Measured, deployed speech
# makes the request a different way -- can-you 18.8% and please 18.3% for
# Help_FindMyHearingAids against a bare imperative at 3.1%. Reproducing the RATE
# with the wrong SHAPE has not reproduced the speech, and no spec said so.
#
# Every share below is re-derived from train.csv here, the same standard check
# 74 holds the headline rates to, and named with boundary_lint's own pattern
# names so the spec and the instrument share a vocabulary.
_bad_shape = []
_SHARE = re.compile(r'(can-you|please|imperative|stated-need) (\d+\.?\d*)%')
for _n in sorted(_bl2.carve_out_intents()):
    _line = [x for x in by[_n]['boundary_cases'] if 'measured on the same rows' in x]
    if not _line:
        _bad_shape.append((_n, 'no shape breakdown')); continue
    _dep = _byi.get(_n, [])
    _c = __import__('collections').Counter()
    for _t in _dep:
        _cls, _which = _bl2.surface_form(_t)
        if _cls == 'command-shaped':
            _c[_which] += 1
    for _which, _pct in _SHARE.findall(_line[0]):
        _act = 100 * _c[_which] / len(_dep) if _dep else 0.0
        # Compared at one decimal place, which is what the spec states, rather
        # than with check 74's 0.05 window. 42 of 224 is 18.75%, which the spec
        # writes as 18.8% and a 0.05 tolerance rejects by exactly nothing.
        if round(_act, 1) != round(float(_pct), 1):
            _bad_shape.append((_n, f'{_which} claims {_pct}%, actual {_act:.2f}%'))
    # the breakdown must account for the headline rate, not a subset of it
    _sum = sum(float(p) for _, p in _SHARE.findall(_line[0]))
    _tot = 100 * sum(_c.values()) / len(_dep) if _dep else 0.0
    if abs(_sum - round(_tot, 1)) >= 0.15:
        _bad_shape.append((_n, f'shares sum to {_sum:.1f}%, headline is {_tot:.1f}%'))
add(f'131 D24 every carve-out states its shape breakdown, and every share re-derives {_bad_shape}',
    not _bad_shape)
add('132 D24 and each says the rate alone is not enough',
    all(any('the wrong SHAPE has not reproduced the speech' in x
            for x in by[_n]['boundary_cases'])
        for _n in _bl2.carve_out_intents()))

# --- D25, the sentence that may have primed what it warned against -------
# D24's line ended by NAMING the explain-request trigger words ("show me, tell
# me and help me"). After it, Help_FindMyHearingAids' explain-request share went
# 32.0% -> 54.5% against a deployed 12.9%, on the one intent that did not
# improve -- while Help_Pairing went 21.7% -> 16.0%. n is 22 and 25, so that is
# a signal and not a proof, but it points the wrong way on exactly the intent
# that was still failing. The words are gone; a measured ceiling replaces them.
_bad_ex = []
_EX = re.compile(r'explain-request share for this intent is (\d+\.?\d*)%')
for _n in sorted(_bl2.carve_out_intents()):
    _txt = ' '.join(by[_n]['boundary_cases'])
    _m = _EX.search(_txt)
    if not _m:
        _bad_ex.append((_n, 'no explain-request ceiling')); continue
    _rows = _byi.get(_n, [])
    _act = 100 * sum(1 for _t in _rows
                     if _bl2.surface_form(_t)[0] == 'explain-request') / len(_rows)
    if round(_act, 1) != round(float(_m.group(1)), 1):
        _bad_ex.append((_n, f'claims {_m.group(1)}%, actual {_act:.2f}%'))
add(f'133 D25 every carve-out states a re-derived explain-request ceiling {_bad_ex}',
    not _bad_ex)
add('134 D25 and none of them names the trigger words',
    not [s_['name'] for s_ in specs for x in s_['boundary_cases']
         if 'show me, tell me and help me' in x])

# --- D26, the number that overstated the gap -----------------------------
# boundary_lint reported 4.0% command-shaped for Help_FindMyHearingAids against
# a deployed 40.6%, which reads as "generation produced almost no direct
# requests". It produced them at about seven tenths of the deployed rate and
# phrased them with a help-verb, which surface_form resolves to explain-request
# before it ever tests the command patterns. Two different defects, one number.
# The pass condition is unchanged -- both sides are measured the same way -- but
# the report now carries the diagnostic beside it.
add('135 D26 boundary_lint separates "no direct requests" from "phrased with a help-verb"',
    callable(getattr(_bl2, 'carries_command', None))
    and _bl2.carries_command('can you help me find my hearing aids')
    # ...and that same row is NOT command-shaped, which is the whole point
    and _bl2.surface_form('can you help me find my hearing aids')[0] == 'explain-request'
    and 'Cmd-patterned' in open(f'{D}/boundary_lint.py').read())
add('136 D26 and it is reported, never scored',
    'DIAGNOSTIC, not scored' in open(f'{D}/boundary_lint.py').read()
    and 'rec["cmd"]' in open(f'{D}/boundary_lint.py').read())

# --- D27, marker-free Help (verification spec section 12.C) --------------
# "Are there Help examples that contain none of how/what/where/can I/is there?
#  If not, the Help dataset is probably too template-driven."
#
# Measured before it was built, because a check that cannot fire is worse than
# none: 45.9% of the 2,946 deployed Help rows are marker-free, per-intent 19.3%
# to 78.1%. It fires on real data -- Help_Tinnitus generated 8.0% against a
# deployed 55.0% -- and it sees something boundary_lint's other half cannot:
# Help_Tinnitus is 0.0% command-shaped on BOTH sides and passes that section.
_blsrc = open(f'{D}/boundary_lint.py').read()
add('137 D27 marker_free exists and is independent of surface form',
    callable(getattr(_bl2, 'marker_free', None))
    # carries "how" in the middle -- marker-carrying, and an explain-request
    and not _bl2.marker_free('tell me how to pair them')
    and _bl2.surface_form('tell me how to pair them')[0] == 'explain-request'
    # no marker at all, and command-shaped
    and _bl2.marker_free('please find my hearing aids')
    and _bl2.surface_form('please find my hearing aids')[0] == 'command-shaped')
# The list is the specification's, verbatim. Widening it shrinks the marker-free
# set and makes the test quietly stricter than the property it names.
#
# Tested by BEHAVIOUR, not by dissecting the compiled pattern string -- that was
# fragile surgery on an implementation detail, and this file has already been
# fooled twice by inspecting text instead of running the thing.
_M_IN = ['how do i pair', 'what does this do', 'where is the button',
         'can i pair them myself', 'is there a way to pair']
_M_OUT = ['why is it doing that', 'which memory is on', 'when does it charge',
          'please pair my aids', 'tell me about pairing', 'pair my hearing aids']
add(f'138 D27 the marker list is exactly the five the spec names '
    f'{[t for t in _M_IN if _bl2.marker_free(t)] + [t for t in _M_OUT if not _bl2.marker_free(t)]}',
    all(not _bl2.marker_free(t) for t in _M_IN)
    and all(_bl2.marker_free(t) for t in _M_OUT))
# The report must not carry a hardcoded corpus statistic. Five stale numbers have
# been found in this codebase already; a generated report is the worst place for
# a sixth, because it looks freshly measured.
add('138b D27 the overall marker-free share is computed, not written down',
    'overall = f"{_hf / _hn:.1%}"' in _blsrc and '45.9%' not in _blsrc.split('def report_markers')[1])
# The marker regex assumes apostrophes survive normalise ("what's" -> "what s").
# An ASR corpus that drops them spells it "whats", which \bwhat\b does not match:
# such a row would be counted marker-FREE, inflating the deployed baseline and
# making the test stricter than the property it names.
#
# Scoped to Help intents, because that is where the metric applies. Measured
# 2026-09-08: zero in Help. NOT hypothetical though -- the corpus already holds
# four of them, 2 in Cmd.BatteryLevel and 2 in Fallback. The assumption is one
# Help row away from being wrong, which is why it is asserted rather than noted.
_CONTR = re.compile(r"\b(whats|hows|wheres|whos|whens|whys)\b")
_contr_hits = [(_n, _t) for _n, _rs in _byi.items() if _n.startswith('Help')
               for _t in _rs if _CONTR.search(_bl2.normalise(_t))]
add(f'138c D27 no apostrophe-less wh-contraction in Help, which the marker list '
    f'would miss ({len(_contr_hits)} found)', not _contr_hits)
# Lower tail only. Producing MORE marker-free rows is another check's business.
add('139 D27 the test is one-sided on the lower tail, per intent',
    'p_at_most(g["free"], g["n"], b_rate)' in _blsrc
    and 'LOWER tail only' in _blsrc
    and 'b_rate = b["free"] / b["n"]' in _blsrc)
# Scored, not decorative: it has to reach the exit code. Grepping the source for
# 'failures += marker_failures' is NOT enough -- a mutation test walked straight
# through that, because `pass  # failures += marker_failures` contains it too.
# That is the second time a substring check has been fooled in this file. So:
# call the function on a case whose answer is arithmetic, and read the WIRING
# out of the parsed syntax tree rather than the text.
_fail_case = _bl2.report_markers(
    {'Help_X': {'n': 100, 'free': 60}},
    {'Help_X': {'n': 25, 'free': 0, 'flags': 0, 'cmd': 0, 'why': __import__('collections').Counter()}},
)
_pass_case = _bl2.report_markers(
    {'Help_X': {'n': 100, 'free': 60}},
    {'Help_X': {'n': 25, 'free': 15, 'flags': 0, 'cmd': 0, 'why': __import__('collections').Counter()}},
)
_wired = False
for _node in _ast.walk(_ast.parse(_blsrc)):
    if (isinstance(_node, _ast.AugAssign) and isinstance(_node.op, _ast.Add)
            and isinstance(_node.target, _ast.Name) and _node.target.id == 'failures'
            and isinstance(_node.value, _ast.Name) and _node.value.id == 'marker_failures'):
        _wired = True
add('140 D27 the section is scored, and the scoring is really wired to the exit code',
    _fail_case[1] == 1 and _pass_case[1] == 0 and _wired)
# Same noise guard as everywhere else in this file.
add('141 D27 it refuses to judge when deployed predicts fewer than 3 rows',
    'if expected < MIN_FLAGS_TO_FAIL' in _blsrc)

# --- the generated report ------------------------------------------------
md = open(f'{D}/SPEC_REVIEW.md').read()
a = md.split('### 2a')[1].split('### 2b')[0]
add('31 Section 2a empty -- Health/Home now guarded on both sides (DEFERRED E1 closed)',
    'None' in a)
b = md.split('### 2b')[1].split('### 2c')[0]
# 2b is NOT empty and is not expected to be: the one pair below is logged in
# DEFERRED A1 as a queued edit against a deferred spec. Assert the exact known
# state so that a NEW pair appearing here fails the run.
# Was 'holds only the logged EdgeModeDeactivate/MaskMode pair'. D17 closed it
# from both sides -- queued A1 edit 3 -- so 2b is now empty like 2a and 2c.
add('32 Section 2b empty -- the last logged pair closed by D17', '✅ None' in b)
add('32b Section 2c empty', 'both sides' in md.split('### 2c')[1].split('## 3')[0])
# 21, not 23 -- the two messaging pairs were a false contract, see DEFERRED E9.
add('33 Section 3: 0 of 21 pairs failing', '**0 of 21 pairs are not mutual neighbours.**' in md)
# Flipped at sign-off (2026-08-30). Both used to assert the gate was SHUT --
# 18 unticked boxes and REQUIRES HUMAN REVIEW in meta. They now assert it is
# open and properly recorded, so the state cannot drift back unnoticed either
# way. The ticks come from spec_review.SIGNED_OFF, never from the Markdown.
add('34 all 18 sign-off boxes ticked, none left open',
    md.count('☑') == 18 and md.count('☐') == 0 and len(sr.SIGNED_OFF) == 18
    and 'Akash Rawat' in md)
add('35 REQUIRES HUMAN REVIEW gone, replaced by a sign_off record',
    'REQUIRES HUMAN REVIEW' not in str(I.get('meta'))
    and I['meta']['sign_off']['reviewed_by'] == 'Akash Rawat'
    and I['meta']['sign_off']['date'] == '2026-08-30'
    and 'D1-D18' in I['meta']['sign_off']['record'])
# bootstrap_specs.py WRITES that meta block, so editing intent_specs.yaml alone
# would have been reverted by the next regeneration. Assert the source agrees.
# Was a presence check only -- '"sign_off"' appears in the source, done. That
# could not see a drifting VALUE, and one had already drifted: meta.sign_off.scope
# said one thing in the yaml and another in the source that regenerates it, so
# the next bootstrap run would have erased D20's qualifier. Now compares every
# field of the block against the literal the source would emit.
_bs = open(f'{D}/bootstrap_specs.py').read()
_blk = re.search(r'"sign_off": \{(.*?)\n            \},', _bs, re.S)
_src = {}
if _blk:
    for _k in ('reviewed_by', 'date', 'scope', 'record', 'method'):
        _m = re.search(rf'"{_k}": \(\n((?:\s*".*?"\n)+)\s*\)', _blk.group(1))
        if _m:
            _src[_k] = ''.join(_ast.literal_eval(_l.strip()) for _l in _m.group(1).splitlines())
        else:
            _m = re.search(rf'"{_k}": "(.*?)"', _blk.group(1))
            if _m:
                _src[_k] = _m.group(1)
add(f'35b bootstrap_specs.py would REGENERATE the same sign_off block '
    f'{sorted(k for k in _src if _src[k] != I["meta"]["sign_off"].get(k))}',
    bool(_blk) and len(_src) == 5 and _src == dict(I['meta']['sign_off'])
    and 'REQUIRES HUMAN REVIEW' not in _bs)
add('35c authored_specs.yaml no longer claims review is pending',
    'pending human review' not in open(f'{D}/authored_specs.yaml').read()
    and 'Human review completed 2026-08-30' in open(f'{D}/authored_specs.yaml').read())
# FOUND WHILE WIRING THE SIGN-OFF. authored_by drifted between the two files on
# Cmd.VolumeIncrease and nothing noticed, because the drift guard covers the 7
# CONTENT fields only. bootstrap_specs.py reads authored_by to build
# _provenance, so a regeneration would have silently changed the provenance
# tally. Now guarded.
_A = {s['name']: s for s in yaml.safe_load(open(f'{D}/authored_specs.yaml'))['intents']}
add('35d authored_by matches _provenance.model on all 57',
    not [n for n, s in by.items()
         if _A[n].get('authored_by') != s.get('_provenance', {}).get('model')])
add('35e provenance tally matches the specs it describes',
    I['meta']['sources'] == {'assistant-session (claude-opus-5)': 56,
                             'human (from blueprint, for privacy)': 1})

for l, v in R: print(f'{l:<56}{"PASS" if v else "FAIL"}')
n = sum(1 for _, v in R if v)
print(f'\n{n}/{len(R)} pass | {sum(len(s["neighbor_intents"]) for s in specs)} neighbour links')
sys.exit(0 if n == len(R) else 1)
