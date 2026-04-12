import nbformat as nbf
import warnings
warnings.filterwarnings('ignore')

with open('saab_deep_pipeline.py', 'r') as f:
    code = f.read()

nb = nbf.v4.new_notebook()

# Split the code block roughly into sections
import re
sections = re.split(r'# ─── (.*?) ────────────────────────────────────────────────────────────', code)
cells = []

# Title block
title_md = """# SAAB-DL: SMOTE-Aware Adaptive Boosting with Deep Learning Feature Fusion

## Hybrid Deep Learning Fraud Detection

This notebook combines a custom "SMOTE-Aware Adaptive Boosting" (SAAB) algorithm with deep learning feature extraction techniques (VAE and GNN) to supersede existing tree-based models and published research.
"""
cells.append(nbf.v4.new_markdown_cell(title_md))

# Collect imports up to main()
pre_main_code = code.split('def main():')[0]
cells.append(nbf.v4.new_code_cell(pre_main_code))

main_func = code.split('def main():')[1].split('if __name__ ==')[0]

# Rough splitting logic for the main function into cells
chunks = re.split(r'# ─── (.*?) ────────────────────────────────────────────────────────*?', main_func)

for i in range(1, len(chunks), 2):
    header = chunks[i].strip()
    content = chunks[i+1].strip()
    # Un-indent main content
    unindented = '\n'.join([line[4:] if line.startswith('    ') else line for line in content.split('\n')])
    cells.append(nbf.v4.new_markdown_cell(f"## {header}"))
    cells.append(nbf.v4.new_code_cell(unindented))

nb['cells'] = cells
with open('SAAB_Deep_Learning_Fraud_Detection.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Notebook created.')
