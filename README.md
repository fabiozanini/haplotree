# haplotree
Haplotype tree viewer

- **Authors**: Fabio Zanini & Richard Neher
- **License**: MIT

## Usage
```bash
python src/treeviewer.py data/tree_hap.newick
python src/treeviewer.py --help
```
Compatible with both `python2` and `python3`.

## Requirements
```python
from Bio import Phylo
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```
