"""Fills missing DOI fields in references.bib from the Crossref API (needs internet).
Usage: python add_dois.py references.bib > references_with_doi.bib
UNTESTED against the live API from my sandbox (network was blocked): check every DOI it adds
against the publisher page, and ignore matches with a low title score."""
import re, sys, json, urllib.parse, urllib.request, difflib
bib = open(sys.argv[1], encoding='utf-8').read()
def lookup(title):
    q = urllib.parse.urlencode({'query.bibliographic': title, 'rows': 1})
    with urllib.request.urlopen('https://api.crossref.org/works?' + q, timeout=30) as r:
        it = json.load(r)['message']['items'][0]
    t = (it.get('title') or [''])[0]
    return it.get('DOI'), difflib.SequenceMatcher(None, title.lower(), t.lower()).ratio()
out = []
for entry in re.split(r'(?=^@)', bib, flags=re.M):
    m = re.search(r'title\s*=\s*[{"](.+?)[}"],?\s*\n', entry, flags=re.S)
    if m and 'doi' not in entry.lower():
        try:
            doi, score = lookup(re.sub(r'[{}]', '', m.group(1)))
            if doi and score > 0.85:
                entry = re.sub(r'\n}\s*$', f',\n  doi = {{{doi}}}\n}}\n', entry.rstrip()) + '\n'
            else:
                sys.stderr.write(f'no confident match ({score:.2f}): {m.group(1)[:70]}\n')
        except Exception as e:
            sys.stderr.write(f'lookup failed: {e}\n')
    out.append(entry)
print(''.join(out))
