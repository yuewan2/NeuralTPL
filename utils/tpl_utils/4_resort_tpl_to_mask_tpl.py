import pandas as pd
from test_mask_tpl import *
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    uspto_50k = pd.read_csv('../dataset_gln/uspto-50k_tpl_modified.csv')
    index_list = uspto_50k['templates'].index.tolist()
    prod_smiles = uspto_50k['prod_smiles'].tolist()
    templates = uspto_50k['templates'].tolist()
    resort_templates = uspto_50k['resort_template'].tolist()

    mask_templates = []
    mask_templates_rt_oly_lvgp = []
    for i, tpl in tqdm(enumerate(resort_templates),total=len(resort_templates)):
        try:
            if tpl is not np.nan:
                mask_tpl = mask_group(tpl, mask_type='leaving group')
                mask_templates.append(mask_tpl)
                mask_tpl_o_lvgp = mask_group(tpl, mask_type='maped group')
                mask_templates_rt_oly_lvgp.append(mask_tpl_o_lvgp)
            else:
                mask_templates.append(None)
                mask_templates_rt_oly_lvgp.append(None)
        except Exception as e:
            print(e)
            print(i)
            print(tpl)
            break

    uspto_50k['mask_template'] = mask_templates
    uspto_50k['mask_template_only_leaving_group'] = mask_templates_rt_oly_lvgp
    uspto_50k.to_csv('../dataset_gln/uspto-50k_tpl_modified.csv', index=False)
    # [c:5]:[c;H0;D3;+0:4](:[c:6])-[c;H0;D3;+0:1](:[c:2]):[c:3]>>Br-[c;H0;D3;+0:1](:[c:2]):[c:3].B1(-[c;H0;D3;+0:4](:[c:5]):[c:6])-O-C(-C)(-C)-C(-C)(-C)-O-1
