import nbformat as nbf
import re

with open('SAAB_Deep_Learning_Fraud_Detection.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

# Strip all instances of syntax errors caused by random UI lines '───' being parsed as code
for cell in nb.cells:
    if cell.cell_type == 'code':
        # Remove any line that is just dashes `─` or `print(...)` headers that aren't quite right
        new_lines = []
        for line in cell.source.split('\n'):
            if re.match(r'^[─]+$', line.strip()):
                continue
            new_lines.append(line)
        cell.source = '\n'.join(new_lines)
            
with open('SAAB_Deep_Learning_Fraud_Detection.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Notebook fixed again.')
