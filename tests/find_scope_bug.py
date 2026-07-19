"""Find scope variable reference causing NameError in visualize_results.py"""
lines = open('4.SemanticMapping/visualize_results.py', 'r').readlines()
for i, l in enumerate(lines):
    stripped = l.split('#')[0]
    if 'scope' in stripped and 'SCOPE' not in stripped and 'scope_' not in stripped:
        print(f"{i+1}: {l.rstrip()[:150]}")
