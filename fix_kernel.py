import nbformat as nbf

with open('SAAB_Deep_Learning_Fraud_Detection.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

if 'kernelspec' in nb.metadata:
    nb.metadata.kernelspec['display_name'] = 'Python 3.12'
    nb.metadata.kernelspec['name'] = 'python312'
    
with open('SAAB_Deep_Learning_Fraud_Detection.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Kernel updated.')
