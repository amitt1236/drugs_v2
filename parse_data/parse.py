from lxml import etree as ET
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import collections

def collapse_list_values(row):
    for key, value in row.items():
        if isinstance(value, list):
            row[key] = '|'.join(value)
    return row

# load xml
tree = ET.parse("/Users/amitaflalo/Desktop/drugs_v1/parse_data/drugs.xml")
root = tree.getroot()

# parse data
ns = '{http://www.drugbank.ca}'
inchi  = "{ns}calculated-properties/{ns}property[{ns}kind='InChI']/{ns}value"
counter = 0
rows = list()
count = 0
for drug in tqdm(root):
    assert drug.tag == ns + 'drug'
    
    groups = [group.text for group in drug.findall("{ns}groups/{ns}group".format(ns = ns))]
    ind = [ind.text for ind in drug.findall("{ns}indication".format(ns = ns))]

    if (ind[0] is None) or groups is None:
        continue
    
    ind = ind[0].split('.')[0]

    if (not any("approved" in s for s in groups)) or drug.findtext(inchi.format(ns = ns)) is None:
        continue

    try:
        mol = Chem.MolFromInchi(drug.findtext(inchi.format(ns = ns)))
    except:
        continue

    row = collections.OrderedDict()
    
    row['drugbank_id'] = drug.findtext(ns + "drugbank-id[@primary='true']")
    row['name'] = drug.findtext(ns + "name")
    row['groups'] = groups
    row['indication'] = ind
    row['inchi'] = drug.findtext(inchi.format(ns = ns))
    row['type'] = drug.get('type')

    rows.append(row)

    
rows = list(map(collapse_list_values, rows))

columns = ['drugbank_id', 'name', 'groups', 'indication', 'inchi', 'type']
drugbank_df = pd.DataFrame.from_dict(rows)[columns]
# print(drugbank_df['indication'])


drugbank_slim_df = drugbank_df[
    drugbank_df.groups.map(lambda x: 'approved' in x) &
    drugbank_df.inchi.map(lambda x: x is not None) &
    drugbank_df.type.map(lambda x: x == 'small molecule') &
    drugbank_df.indication.map(lambda x: x is not None)
]

print(len(drugbank_slim_df))
drugbank_slim_df.to_csv('drugs_all.csv')

# for index, row in drugbank_df.iterrows():
    
#     print(row['inchikey'])
#     input("Press Enter to continue...\n\n\n\n")