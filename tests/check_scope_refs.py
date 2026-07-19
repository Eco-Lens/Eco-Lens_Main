"""Check for bare {scope} in f-strings that would cause NameError"""
lines = open('4.SemanticMapping/visualize_results.py', 'r').readlines()
for i, l in enumerate(lines):
    # Look for f-strings with {scope} but not {scope_} or SCOPE or .get
    if '{scope}' in l and '{scope_' not in l and 'SCOPE' not in l and 'bp.get' not in l and 't.get' not in l and 'b.get' not in l:
        print(f'WARNING line {i+1}: {l.rstrip()[:150]}')
print('Done checking.')
