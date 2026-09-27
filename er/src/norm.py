"""Text normalization for business names and addresses (country-aware, open-set countries).

Two layers:
  clean_*  : unicode + punctuation clean-up, keeps native-script (Indic) tokens as-is
  norm_*   : maps native tokens via a learned dict (dicts.py), transliterates the rest,
             canonicalizes legal forms / street types / states, parses address parts.
"""
import os
import re
import json
import unicodedata
from unidecode import unidecode

# ---------------------------------------------------------------- scripts
IND_RE = re.compile(r'[ऀ-෿]')          # Devanagari .. Malayalam (+ Sinhala edge)
ZW_RE = re.compile(r'[​-‍﻿]')


def has_ind(s):
    return bool(IND_RE.search(s))


# ---------------------------------------------------------------- lexicons
LEGAL = {
    'pvt': 'private', 'pvtltd': 'private limited', 'private': 'private', 'ltd': 'limited', 'limited': 'limited',
    'ltda': 'limited', 'llp': 'llp', 'llc': 'llc', 'pllc': 'pllc', 'lllp': 'lllp', 'lp': 'lp', 'inc': 'inc',
    'incorporated': 'inc', 'corp': 'corp', 'corporation': 'corp', 'co': 'co', 'company': 'co', 'cos': 'co',
    'plc': 'plc', 'pc': 'pc', 'pa': 'pa', 'opc': 'opc', 'public': 'public', 'sarl': 'sarl', 'sas': 'sas',
    'sasu': 'sasu', 'sa': 'sa', 'sci': 'sci', 'eurl': 'eurl', 'snc': 'snc', 'scop': 'scop', 'selarl': 'selarl',
    'cie': 'co', 'gmbh': 'gmbh', 'ag': 'ag', 'bv': 'bv', 'nv': 'nv',
}
STOP = {'the', 'and', 'of', 'a', 'an', 'de', 'du', 'des', 'la', 'le', 'les', 'et', 'l', 'd', 'ms', 'm/s'}
TITLE = {'smt', 'shri', 'sri', 'shree', 'mr', 'mrs', 'dr', 'm', 'mme', 'ste', 'st'}
FKA_RE = re.compile(r'\b(?:formerly known as|formerly|f/k/a|fka|d/b/a|dba|doing business as|trading as|t/a|a/k/a|aka)\b')
DOM_RE = re.compile(r'^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-\.]*?)\.(?:com|in|net|org|co|fr|io|biz|info|us|co\.in|org\.in|net\.in|eu)$')
LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't', '@': 'a', '$': 's'})

STREET_US = {
    'st': 'street', 'str': 'street', 'ave': 'avenue', 'av': 'avenue', 'avn': 'avenue', 'rd': 'road', 'dr': 'drive',
    'drv': 'drive', 'ln': 'lane', 'blvd': 'boulevard', 'ct': 'court', 'crt': 'court', 'cir': 'circle',
    'pl': 'place', 'trl': 'trail', 'ter': 'terrace', 'terr': 'terrace', 'hwy': 'highway', 'pkwy': 'parkway',
    'expy': 'expressway', 'fwy': 'freeway', 'sq': 'square', 'mt': 'mount', 'ft': 'fort', 'pt': 'point',
    'cv': 'cove', 'xing': 'crossing', 'aly': 'alley', 'bnd': 'bend', 'hl': 'hill', 'hls': 'hills',
    'holw': 'hollow', 'lp': 'loop', 'mdw': 'meadow', 'mdws': 'meadows', 'pass': 'pass', 'pike': 'pike',
    'plz': 'plaza', 'rdg': 'ridge', 'run': 'run', 'spg': 'spring', 'spgs': 'springs', 'tpke': 'turnpike',
    'vw': 'view', 'vly': 'valley', 'vis': 'vista', 'way': 'way', 'wy': 'way', 'cres': 'crescent',
    'crk': 'creek', 'est': 'estate', 'ests': 'estates', 'grv': 'grove', 'hts': 'heights', 'isl': 'island',
    'jct': 'junction', 'lk': 'lake', 'lks': 'lakes', 'mnr': 'manor', 'orch': 'orchard', 'pnes': 'pines',
    'shr': 'shore', 'sta': 'station', 'trce': 'trace', 'trak': 'track', 'vlg': 'village', 'rte': 'route',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west', 'ne': 'northeast', 'nw': 'northwest',
    'se': 'southeast', 'sw': 'southwest', 'apt': 'unit', 'apartment': 'unit', 'ste': 'unit', 'suite': 'unit',
    'unit': 'unit', 'fl': 'floor', 'flr': 'floor', 'bldg': 'building', 'rm': 'room',
}
STREET_IN = {
    'rd': 'road', 'st': 'street', 'opp': 'opposite', 'nr': 'near', 'hno': 'house', 'h': 'house',
    'bldg': 'building', 'apts': 'apartment', 'apt': 'apartment', 'fl': 'floor', 'flr': 'floor',
    'gf': 'ground floor', 'ff': 'first floor', 'sf': 'second floor', 'nagar': 'nagar', 'ngr': 'nagar',
    'mkt': 'market', 'marg': 'marg', 'sec': 'sector', 'sect': 'sector', 'dist': 'district', 'tal': 'taluk',
    'tq': 'taluk', 'po': 'post', 'ps': 'police station', 'vill': 'village', 'vpo': 'village post',
    'extn': 'extension', 'ext': 'extension', 'indl': 'industrial', 'ind': 'industrial', 'estt': 'estate',
    'no': '', 'num': '', 'door': '', 'plot': 'plot', 'shop': 'shop', 'khno': 'khasra', 'kh': 'khasra',
    'bombay': 'mumbai', 'bangalore': 'bengaluru', 'calcutta': 'kolkata', 'madras': 'chennai',
    'gurgaon': 'gurugram', 'poona': 'pune', 'ahmadabad': 'ahmedabad', 'baroda': 'vadodara',
}
STREET_FR = {
    'r': 'rue', 'rue': 'rue', 'av': 'avenue', 'ave': 'avenue', 'bd': 'boulevard', 'bld': 'boulevard',
    'blvd': 'boulevard', 'boul': 'boulevard', 'pl': 'place', 'all': 'allee', 'che': 'chemin', 'chem': 'chemin',
    'ch': 'chemin', 'imp': 'impasse', 'rte': 'route', 'rt': 'route', 'sq': 'square', 'fg': 'faubourg',
    'fbg': 'faubourg', 'crs': 'cours', 'qu': 'quai', 'qua': 'quai', 'res': 'residence', 'lot': 'lotissement',
    'lt': 'lotissement', 'st': 'saint', 'ste': 'sainte', 'pte': 'porte', 'prom': 'promenade', 'sen': 'sente',
    'hle': 'halle', 'za': 'zone', 'zi': 'zone', 'zac': 'zone', 'mte': 'montee', 'pass': 'passage',
    'bis': 'bis', 'ter': 'ter', 'gal': 'general', 'gen': 'general', 'mal': 'marechal', 'mar': 'marechal',
    'pdt': 'president', 'prof': 'professeur', 'cdt': 'commandant', 'lt': 'lieutenant', 'cte': 'comte',
    'no': '', 'num': '', 'numero': '', 'ndeg': '', 'psg': 'passage', 'appt': 'appartement', 'app': 'appartement',
    'bat': 'batiment', 'bt': 'batiment', 'esc': 'escalier', 'rdc': 'rez de chaussee',
}
US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california', 'co': 'colorado',
    'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
    'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana',
    'me': 'maine', 'md': 'maryland', 'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota',
    'ms': 'mississippi', 'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york', 'nc': 'north carolina',
    'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania',
    'ri': 'rhode island', 'sc': 'south carolina', 'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas',
    'ut': 'utah', 'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
    'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia', 'pr': 'puerto rico',
}
IN_STATES = {
    'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam', 'br': 'bihar', 'cg': 'chhattisgarh',
    'ct': 'chhattisgarh', 'ga': 'goa', 'gj': 'gujarat', 'hr': 'haryana', 'hp': 'himachal pradesh',
    'jh': 'jharkhand', 'ka': 'karnataka', 'kl': 'kerala', 'mp': 'madhya pradesh', 'mh': 'maharashtra',
    'mn': 'manipur', 'ml': 'meghalaya', 'mz': 'mizoram', 'nl': 'nagaland', 'od': 'odisha', 'or': 'odisha',
    'pb': 'punjab', 'rj': 'rajasthan', 'sk': 'sikkim', 'tn': 'tamil nadu', 'tg': 'telangana',
    'ts': 'telangana', 'tr': 'tripura', 'up': 'uttar pradesh', 'uk': 'uttarakhand', 'ut': 'uttarakhand',
    'wb': 'west bengal', 'dl': 'delhi', 'jk': 'jammu and kashmir', 'ch': 'chandigarh', 'py': 'puducherry',
    'la': 'ladakh', 'an': 'andaman and nicobar islands', 'dn': 'dadra and nagar haveli', 'ld': 'lakshadweep',
    'orissa': 'odisha', 'uttaranchal': 'uttarakhand', 'pondicherry': 'puducherry',
}
NULLS = {'null', 'none', 'nan', 'na', 'n/a', 'nil', 'unknown'}

# France (test-only country): departments -> region, so 'Bordeaux, Gironde' and 'Bordeaux, Nouvelle-Aquitaine' agree
# (the same role US_STATES / IN_STATES play). Keys are hyphen-free, accent-free whole address components.
FR_REGIONS = {'hauts de france', 'nouvelle aquitaine', 'pays de la loire', 'ile de france', 'bretagne', 'normandie',
              'grand est', 'occitanie', 'provence alpes cote d azur', 'auvergne rhone alpes', 'centre val de loire',
              'bourgogne franche comte', 'corse'}
_FR_DEP = {
    'hauts de france': 'nord|pas de calais|aisne|oise|somme',
    'nouvelle aquitaine': 'gironde|charente|charente maritime|correze|creuse|deux sevres|dordogne|landes|'
                          'lot et garonne|pyrenees atlantiques|haute vienne',
    'pays de la loire': 'loire atlantique|maine et loire|mayenne|sarthe|vendee',
    'ile de france': 'seine et marne|yvelines|essonne|hauts de seine|seine saint denis|val de marne|val d oise',
    'bretagne': 'finistere|cotes d armor|ille et vilaine|morbihan',
    'normandie': 'calvados|eure|orne|seine maritime',
    'grand est': 'ardennes|aube|haute marne|meurthe et moselle|meuse|moselle|bas rhin|haut rhin|vosges',
    'occitanie': 'ariege|aude|aveyron|gard|haute garonne|gers|herault|lozere|hautes pyrenees|pyrenees orientales|'
                 'tarn|tarn et garonne',
    'provence alpes cote d azur': 'alpes de haute provence|hautes alpes|alpes maritimes|bouches du rhone|vaucluse',
    'auvergne rhone alpes': 'ain|allier|ardeche|cantal|drome|isere|haute loire|puy de dome|savoie|haute savoie',
    'centre val de loire': 'cher|eure et loir|indre|indre et loire|loir et cher|loiret',
    'bourgogne franche comte': 'cote d or|doubs|jura|nievre|haute saone|saone et loire|yonne|territoire de belfort',
    'corse': 'corse du sud|haute corse'}
FR_DEPT = {d: r for r, ds in _FR_DEP.items() for d in ds.split('|')}      # city-named departments (paris...) left out
FR_LEGAL = {'ei': 'ei', 'eirl': 'eirl', 'gie': 'gie', 'scp': 'scp', 'sca': 'sca', 'scs': 'scs', 'sep': 'sep',
            'sem': 'sem', 'selas': 'selas', 'selafa': 'selafa', 'sarlu': 'sarlu'}
FR_NAME = {'ets': 'etablissements', 'etabl': 'etablissements', 'cie': 'compagnie', 'assoc': 'association',
           'asso': 'association', 'st': 'saint', 'ste': 'sainte'}
FR_ORD_RE = re.compile(r'(?<=\d)(er|ere|eme|e)$')


def street_map(ctry):
    c = (ctry or '').lower()
    if c == 'us':
        return STREET_US, US_STATES
    if c == 'india':
        return STREET_IN, IN_STATES
    if c == 'france':
        return STREET_FR, {}
    return {}, {}


# ---------------------------------------------------------------- basic clean
def base(s):
    if s is None or s != s:
        return ''
    s = unicodedata.normalize('NFKC', str(s))
    s = ZW_RE.sub('', s)
    return s.strip()


def latin(tok):
    """Latin transliteration + accent strip; lowercase."""
    return unidecode(tok).lower()


def fix_leet(t):
    # only for alpha-dominant name tokens: agr0 -> agro, live1y -> lively
    if not t.isalpha() and any(c.isalpha() for c in t) and sum(c.isdigit() for c in t) <= 2 \
            and sum(c.isalpha() for c in t) >= 3:
        return t.translate(LEET)
    return t


def split_name(s):
    """-> (main, alt, domain_stem). Removes phones/junk; keeps native script."""
    s = base(s).lower()
    dom = ''
    # website attached: 'foo inc | www.foo.com'
    parts = [p.strip() for p in re.split(r'\s[|]\s|\s*\|\s*', s) if p.strip()]
    keep = []
    for p in parts:
        m = DOM_RE.match(p.replace(' ', ''))
        if m and ('.' in p) and ' ' not in p.strip():
            dom = dom or m.group(1).replace('.', '').replace('-', '')
        else:
            keep.append(p)
    s = ' '.join(keep)
    alt = ''
    m = FKA_RE.search(s)
    if m:
        s, alt = s[:m.start()], s[m.end():]
    return s, alt, dom


def clean_name_toks(s, fr=False):
    """tokens of a name, punctuation stripped, native-script tokens kept."""
    s = s.replace('&', ' and ').replace('+', ' and ')
    if fr:                                               # French elisions: l'atelier -> l atelier (l/d are stop words)
        s = re.sub(r"['’]", ' ', s)
    s = re.sub(r'\d[\d\s\-]{6,}\d', ' ', s)                 # phone numbers
    s = re.sub(r"(?<=[\w\u0900-\u0DFF])[.'’](?=[\w\u0900-\u0DFF])", '', s)                 # l.l.c -> llc, o'neil -> oneil
    s = re.sub(r"[^\w\s\u0900-\u0DFF]", ' ', s)
    return [t for t in s.split() if t]


def clean_addr(s):
    s = base(s).lower()
    s = s.replace('<null>', ' ').replace('#', ' ')
    return s


# ---------------------------------------------------------------- normalize with dict
def map_tok(t, tdict):
    if has_ind(t):
        v = tdict.get(t) if tdict else None
        if v is not None:
            return v, 1
        return latin(t).replace("'", ''), 2          # unseen native token -> rough latin
    return latin(t), 0


def norm_name(raw, tdict=None, ctry=None):
    """-> dict(nm, core, legal, alt, dom, nind, nunk)."""
    fr = ctry == 'France'
    lg = {**LEGAL, **FR_LEGAL} if fr else LEGAL
    s, alt, dom = split_name(raw)
    out, legal, nind, nunk = [], [], 0, 0
    for t in clean_name_toks(s, fr):
        v, k = map_tok(t, tdict)
        nind += k > 0
        nunk += k == 2
        for w in v.split():
            w = fix_leet(w)
            out.append(FR_NAME.get(w, w) if fr else w)
    # multi-token legal: 'l l c' , 'p v t'
    j = ' '.join(out)
    j = re.sub(r'\b(l l c|l l p|p l l c|s a s|s a r l|s a|p v t|l t d|i n c)\b', lambda m: m.group(0).replace(' ', ''), j)
    toks = j.split()
    core = []
    for w in toks:
        if w in lg:
            legal.append(lg[w])
        elif w not in STOP:
            core.append(w)
    nm = ' '.join(lg.get(w, w) for w in toks if w not in STOP)
    alt_t = []
    if alt:
        for t in clean_name_toks(alt, fr):
            alt_t.append(map_tok(t, tdict)[0])
    if not core and dom:
        core = [dom]
    return dict(nm=nm, core=' '.join(core), legal=' '.join(sorted(set(' '.join(legal).split()))),
                alt=' '.join(alt_t), dom=dom, nind=nind, nunk=nunk)


NUM_RE = re.compile(r'\d+')


def _load_city_region():
    """FR_CITY_REGION: city -> region, learned only from this country's own S1 records (dicts.learn_city_region();
    additive fallback for norm_addr() below, applied only when the department/region table found nothing -- see
    W/fr_city_region.json). OFF BY DEFAULT (env ER_CITY_REGION=1 to enable): norm.py feeds prep -> every n_* parquet
    -> every downstream feature, so flipping this on changes what a full run_all.sh reproduces. fin/fin2/fin3/fin4/
    b34c and the running ceq were all built with this off; turn it on only for a build that is meant to include it,
    and say so in Documentation.md. Guarded: also returns {} if the file is absent (before learn_city_region() has
    been run, or on a machine with no ER_WORK data yet)."""
    if os.environ.get('ER_CITY_REGION') != '1':
        return {}
    try:
        w = os.environ.get('ER_WORK', '/scratch/m25csa019/er_work')
        return json.load(open(f'{w}/fr_city_region.json')).get('France', {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


FR_CITY_REGION = _load_city_region()


def norm_num(t, fr=False):
    """house-number canon: strip leading zeros, drop ordinal suffix, keep letter suffix."""
    t = re.sub(r'(?<=\d)(st|nd|rd|th)$', '', t)
    if fr:
        t = FR_ORD_RE.sub('', t)                         # 1er, 2e, 3eme
    return NUM_RE.sub(lambda m: str(int(m.group(0))), t)


def norm_addr(raw, ctry, tdict=None):
    """-> dict(ad, hn, pc, state, comps). hn = numeric tokens (canon), pc = 5/6-digit postal codes."""
    s = clean_addr(raw)
    fr = ctry == 'France'
    if fr:                                               # 'N°24' / 'n° 24' -> '24'
        s = re.sub(r'\bn\s*°\s*', ' ', s).replace('°', ' ')
    smap, states = street_map(ctry)
    comps, toks_all, nums, pcs = [], [], [], []
    for comp in s.split(','):
        comp = comp.strip()
        if not comp:
            continue
        comp = re.sub(r"(?<=[a-z])\.(?=[a-z])", '', comp) if not has_ind(comp) else comp
        raw_t = re.split(r'[\s.;:()\[\]{}"]+', comp)
        ct = []
        for t in raw_t:
            t = t.strip("-/'")
            if not t:
                continue
            v, _ = map_tok(t, tdict)
            for w in v.split():
                w = w.strip("-/'")
                if not w or w in NULLS:
                    continue
                if any(c.isdigit() for c in w):
                    for piece in re.split(r'[-/]', w):
                        if not piece:
                            continue
                        pn = norm_num(piece, fr)
                        if pn.isdigit() and len(pn) in (5, 6) and len(piece) in (5, 6) and ctry != 'US':
                            pcs.append(pn)
                        elif pn.isdigit() and len(pn) >= 7:
                            continue                        # phone-like
                        nums.append(pn)
                    ct.append(norm_num(w, fr))
                elif fr:                                  # saint-nazaire -> saint nazaire ; d'espagne -> espagne
                    for part in re.split(r"[-'’]", w):
                        part = smap.get(part, part)
                        if part and len(part) > 1:
                            ct.extend(part.split())
                else:
                    w = smap.get(w, w)
                    if w:
                        ct.extend(w.split())
        if ct:
            c = ' '.join(ct)
            if fr:
                c = FR_DEPT.get(c, c)                     # department -> region
                if c in comps:                            # 'Nord' + 'Hauts-de-France' -> keep one
                    continue
            comps.append(c)
            toks_all.extend(ct)
    # state: last-ish comp that is a known state abbreviation or full name
    state = ''
    full = set(states.values())
    for c in reversed(comps):
        if c in states:
            state = states[c]
            break
        if c in full:
            state = c
            break
    if state:
        comps = [states.get(c, c) for c in comps]
    if fr:
        state = next((c for c in reversed(comps) if c in FR_REGIONS), '')
        if not state:                                    # additive fallback: bare city name, no department/region text at all
            state = next((FR_CITY_REGION[c] for c in reversed(comps) if c in FR_CITY_REGION), '')
    return dict(ad=' , '.join(comps), hn=' '.join(dict.fromkeys(nums)), pc=' '.join(dict.fromkeys(pcs)),
                state=state)


if __name__ == '__main__':
    tests = [('Maure Williams Colombier Inc', '85 Wayne Avenue, Ticonderoga, NY', 'US'),
             ('maurewilliamscolombier.com', 'Wayne Ave, Ticonderoga Townshiip, New York', 'US'),
             ('-- Holloway Peak Inc Seafood', 'GREENSBORO, NC, 19 1/2 STARDUST TRAIL', 'US'),
             ('Chordia + Pagnters - 7306204978', 'हरियाणा, DOOR NO 1038 SECTOR 9, FARIDABAD', 'India'),
             ('एसएस फूड प्राइवेट लिमिटेड', 'AF-0684, NANDGRAM NEAR MOTHER INDIA PUBLIC SCHOOL. PH. 989, GHAZIABAD, 9487203, उत्तर प्रदेश', 'India'),
             ('BETH EMANUEL OF ACTON INC | www.bethemanu.com', '06606 DAYLA CT, GRANBURY, TX', 'US'),
             ('Onyxlum formerly known as Rocky Committee', '216 Phoenix St, <NULL>, Payson, Arizona', 'US'),
             ('Sri Creative Ágr0 Pvt Ltd', 'Door No 2913 A Ward, Wangi, Bol Kolhapur, Karvir, Kolhapur, MH', 'India'),
             ('Marina Ecole France Sarl', '63 R. DE DIEPPE, LILLE, Hauts-de-France', 'France'),
             ('Dahlia Power Reliable', 'KANSAS CITY, MO, 630 45ND TERRACE, null', 'US'),
             ('Deoliveira, Kincaid & Cole L.L.C.', '5708 Wallace Avenue, Sacramento, CA', 'US')]
    for n, a, c in tests:
        print(norm_name(n), '\n   ', norm_addr(a, c))
