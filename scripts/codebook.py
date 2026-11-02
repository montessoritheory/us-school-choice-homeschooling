"""Parse NHES public-use codebook text (pdftotext -layout) into {var: {label, question, codes{code:label}}}."""
import re, json, sys
def parse(path):
    lines = open(path, encoding='utf8', errors='ignore').read().splitlines()
    out = {}; cur = None; in_tab = False
    for ln in lines:
        m = re.match(r'^Item(?: [^:]*)?:\s*([A-Z0-9_]+)\s*$', ln.strip())
        if m:
            cur = m.group(1); out[cur] = {'label': '', 'question': '', 'codes': {}}; in_tab = False; continue
        if cur is None: continue
        s = ln.strip()
        if s.startswith('Label:'): out[cur]['label'] = s[6:].strip(); continue
        if s.startswith('Question:'): out[cur]['question'] = s[9:].strip(); continue
        if re.match(r'^Code\s+Response', s): in_tab = True; continue
        if in_tab:
            if s.startswith('Total') : in_tab = False; continue
            m = re.match(r'^(-?\d+)\s{2,}(.+?)\s{2,}[\d,]+\s+[\d.]+', s)
            if m: out[cur]['codes'][m.group(1)] = m.group(2).strip()
            elif s and out[cur]['codes'] and not re.match(r'^[\d,.\s]+$', s) and len(s) < 60 and not s[0].isdigit():
                # continuation of a long response label
                k = list(out[cur]['codes'])[-1]
                if not re.search(r'\d', s): out[cur]['codes'][k] += ' ' + s
    return out
if __name__ == '__main__':
    cb = parse(sys.argv[1]); json.dump(cb, open(sys.argv[2], 'w'), indent=1)
    print(len(cb))

def parse2016(path):
    lines = open(path, encoding='utf8', errors='ignore').read().splitlines()
    out = {}; cur = None; in_tab = False
    for ln in lines:
        m = re.match(r'^Variable Name\s*:\s*([A-Z0-9_]+)\s+Label:\s*(.*)$', ln.strip())
        if m:
            cur = m.group(1); lab = re.sub(r'^[\w.]+\.\s+', '', m.group(2).strip()) if re.match(r'^\d', m.group(2).strip()) else m.group(2).strip()
            out[cur] = {'label': lab, 'question': '', 'codes': {}}; in_tab = False; continue
        if cur is None: continue
        s = ln.strip()
        if re.match(r'^Response\s+code', s): in_tab = True; continue
        if in_tab:
            if s.startswith('Total') or s.startswith('____'): in_tab = False; continue
            m = re.match(r'^(.+?)\s{2,}(-?\d+)\s{2,}[\d,]+\s+[\d.]+%', s)
            if m:
                lab = re.sub(r'^-?\d+\s+', '', m.group(1).strip())
                out[cur]['codes'][m.group(2)] = lab
    return out
