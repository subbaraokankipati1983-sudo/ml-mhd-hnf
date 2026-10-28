"""Bibliography sanity checker: required fields, DOI syntax, year range, citation use.
Offline by design; existence/DOI matching was done separately against indexed records."""
import re, sys, csv
bib = open(sys.argv[1] if len(sys.argv) > 1 else 'references.bib').read()
tex = open(sys.argv[2]).read() if len(sys.argv) > 2 else ''
entries = re.findall(r'@(\w+)\{([^,]+),(.*?)\n\}', bib, re.S)
need = {'article': ['author','title','journal','year'], 'inproceedings': ['author','title','booktitle','year'],
        'techreport': ['author','title','institution','year'], 'misc': ['author','title','year']}
rows = []
for typ, key, body in entries:
    f = dict(re.findall(r'(\w+)\s*=\s*[{"](.*?)[}"],?\s*\n', body + '\n'))
    miss = [k for k in need.get(typ, []) if k not in f]
    doi = f.get('doi', '')
    ok_doi = (not doi) or bool(re.fullmatch(r'10\.\d{4,9}/\S+', doi))
    yr = int(f['year']) if f.get('year', '').isdigit() else 0
    used = (f'{key}' in tex) if tex else None
    rows.append([key, typ, 'DOI' if doi else 'no DOI', 'ok' if ok_doi else 'BAD DOI', ','.join(miss) or 'complete',
                 'year ok' if 1900 < yr <= 2026 else 'year?', used])
w = csv.writer(sys.stdout); w.writerow(['key','type','doi','doi_syntax','fields','year','cited']); w.writerows(rows)
