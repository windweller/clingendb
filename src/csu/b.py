import json
import pandas as pd
vet_tc = pd.read_csv('Files_for_parsing/snomed_vet_tc.csv', sep='\t')
final_csu_file = pd.read_csv('final_csu_file2', sep='\t')
combined_desc = pd.read_csv('Files_for_parsing/combined_desc.txt', sep='\t')
def subtype_id_test(x):
    temp_df1 = (vet_tc[['subtype']][vet_tc['supertype'] == x])
    results = temp_df1['subtype']
    return set(results.tolist())

def supertype_id_test(x):
    temp_df1 = (vet_tc[['supertype']][vet_tc['subtype'] == x])
    results = temp_df1['supertype']
    return set(results.tolist())

def id2name(x):
    temp_df1 = (combined_desc[['term']][combined_desc['conceptId'] == x])
    results = temp_df1['term']
    return results.tolist()
clinical_finding = subtype_id_test(404684003)
disease_document = set([105714009, 68843000, 78885002, 344431000009103, 338591000009108, 40733004, 17322007, 723976005, 399981008, 85828009, 414029004, 473010000, 75934005, 363246002, 2492009, 414916001, 363247006, 420134006, 362969004, 271737000, 414022008, 414026006, 362970003, 11888009, 212373009, 262938004, 405538007, 74732009, 313891000009106, 118940003, 50611000119105, 87118001, 362966006, 128127008, 85972008, 49601007, 50043002, 370514003, 422400008, 53619000, 42030000, 362972006, 173300003, 362973001, 404177007, 414032001, 128598002, 105969002, 928000, 111941005, 32895009, 66091009, 414025005, 85983004, 75478009, 77434001, 417163006])
disease_relevant = subtype_id_test(64572001)
out = []
for i, item in enumerate(final_csu_file['all_groups']):
    if i % 1000 == 999: print i
    out_ = set()
    splititem = str(item).replace('S', '').replace('R', '').split('|')
    for number in splititem:
        try:
            number = int(number)
            # out_.add(number)
            out_ = out_ | (supertype_id_test(number) & disease_relevant ) # disease_document)
        except:
            pass
    out.append(list(out_))
hierarchical_levels = pd.DataFrame({'hierarchical_levels': out})
new_final_csu_file = pd.concat([final_csu_file, hierarchical_levels], 1) 
new_final_csu_file.to_csv('final_csu_file_disease_document2', sep='\t', index=False)

