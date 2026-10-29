import json

infile = 'C:/Users/jkarlgre/OneDrive - Advanced Micro Devices Inc/monrepos/yarns-and-loss/experiments/data/_daring_work/out_blimp_06.jsonl'

with open(infile, encoding='utf-8') as f:
    content = f.read()

raw_lines = [l for l in content.split('\n') if l.strip()]

def split_objects(s):
    parts = []
    start = 0
    depth = 0
    in_str = False
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str:
            if ch == '\':
                i += 2
                continue
            if ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    parts.append(s[start:i+1])
        i += 1
    return parts if parts else [s]

good = []
for l in raw_lines:
    parts = split_objects(l)
    good.extend(parts)

print('after split:', len(good))
errs = []
for i, l in enumerate(good):
    try:
        obj = json.loads(l)
        assert 'src' in obj and 'items' in obj and len(obj['items']) == 2
    except Exception as e:
        errs.append((i+1, str(e)[:80]))
print('errors:', errs[:5])

with open(infile, 'w', encoding='utf-8') as f:
    for l in good:
        f.write(l.strip() + '\n')
print('written:', len(good))
print('last src:', json.loads(good[-1])['src'])
