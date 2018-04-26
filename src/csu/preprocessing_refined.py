"""
Process top-level SNOMED code
"""

"""
Takes in raw data and preprocess this into good format
for TorchText to train on
"""

import re
import numpy as np
import json
import pandas as pd
import sys
import copy
from collections import defaultdict

vet_tc = pd.read_csv("../../data/csu/Files_for_parsing/snomed_vet_tc.csv", sep='\t')


def subtype_id_test(x):
    temp_df1 = (vet_tc[['subtype']][vet_tc['supertype'] == x])
    results = temp_df1['subtype']
    return results.tolist()

clinical_finding_code = "404684003"

"""
We train without diagnosis, and with multilabel
"""

np.random.seed(1234)

with open('../../data/csu/Files_for_parsing/other_disease_codes_2000.json', 'r') as f:
    other_disese_codes = json.load(f)
    other_disese_codes_set = set(other_disese_codes)

with open('../../data/csu/Files_for_parsing/non_disease_clinical_finding.json', 'r') as f:
    non_disease_clinical_finding = json.load(f)
    non_disease_clinical_finding_set = set(non_disease_clinical_finding)

with open('../../data/csu/Files_for_parsing/disease_codes_2000.json', 'r') as f:
    disease_codes = json.load(f)

with open('../../data/csu/Files_for_parsing/disease_codes_to_name_2000.json', 'r') as f:
    disease_code_to_name = json.load(f)

# top-SNOMED code
with open('../../data/csu/Files_for_parsing/snomed_code_2000.json', 'r') as f:
    snomed_codes = json.load(f)
    snomed_codes_set = set(snomed_codes)  # we only keep these

with open('../../data/csu/Files_for_parsing/snomed_names_2000.json', 'r') as f:
    snomed_names = json.load(f)

"""
Instead of mapping every SNOMED code and disintegrate the original file
We only identify text that has clinical finding code, 
examine their underlying code, and break them into 3 categories
"""
binned_labels = copy.copy(snomed_codes) + ['Other_Disease', 'Other_Clinical_Finding']
binned_labels_names = copy.copy(snomed_names) + ['Other Disease', 'Other Clinical Finding']

lower_snomed_to_higher_map = defaultdict(str)

for k, v in disease_codes.iteritems():
    if k == '236425005':
        # this is the olde code "Chronic renal disease"
        # we put it as a "subtype" for the newer code
        continue
    binned_labels.append(k)
    binned_labels_names.append(disease_code_to_name[k])
    for sb in v:
        # map list of subtypes to the super type
        lower_snomed_to_higher_map[sb] = k
    # each one should map to themselves as well
    lower_snomed_to_higher_map[k] = k
    if k == '709044004':
        # this is the new "Chronic kidney disease(disorder)"
        # map old code to this new code
        lower_snomed_to_higher_map['236425005'] = k

print("total number of labels/keys after disease and top snomed is: {}".format(len(binned_labels) + 2))

# now we add "other clinical finding"
# and "other diseases"
for ndcf in non_disease_clinical_finding:
    lower_snomed_to_higher_map[ndcf] = 'Other_Clinical_Finding'
for od in other_disese_codes:
    lower_snomed_to_higher_map[od] = 'Other_Disease'

# Ashley threw away many attributes and modifiers
# if a code is not part of the keys, we just throw them out

print("total number of SNOMED code mapped is {}".format(len(lower_snomed_to_higher_map)))

# ======== Split =========
train_size = 0.9

assert (train_size < 1 and train_size > 0)
split_proportions = {
    "train": train_size,
    "valid": (1 - train_size) / 2,
    "test": (1 - train_size) / 2
}
assert (sum([split_proportions[split] for split in split_proportions]) == 1)

print("the data split is: {}".format(split_proportions))


def write_to_tsv(data, file_name, label_list):
    # we are translating labels here
    with open(file_name, 'wb') as f:
        for line in data:
            mapped_labels = [str(label_list.index(l)) for l in line[1].split()]
            f.write(line[0] + '\t' + " ".join(mapped_labels) + '\n')


def count_freq(list_labels):
    dic = defaultdict(int)
    for l in list_labels:
        dic[l] += 1
    return dic


def get_most_freq_label(dic):
    most_f_l = None
    most_f_f = 0.
    for l, f in dic.iteritems():
        if f > most_f_f:
            most_f_l = l
    return most_f_l


def collapse_label(labels):
    list_labels = filter(lambda l: len(l) > 0, labels)
    set_labels = set(list_labels)  # remove redundancies
    return list(set_labels)


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = re.sub(r'^https?:\/\/.*[\r\n]*', '', cleantext, flags=re.MULTILINE)
    return cleantext

def preprocess_text(text, no_description):
    no_html = cleanhtml(text)
    one_white_space = ' '.join(no_html.split())
    no_html_entities = re.sub('&[a-z]+;', '', one_white_space)

    if no_description:
        # delete both diagnosis and discharge status
        no_html_entities = no_html_entities.split('Diagnosis:')[0]

    return no_html_entities


if __name__ == '__main__':
    header = True

    examples = []
    seq_labels_dist = []
    labels_dist = []
    label_num_per_example = []
    total_text = 0

    other_disease_cnt = 0
    other_clinical_f_cnt = 0
    both_cnt = 0

    empty_seq_label_cnt = 0
    with open("../../data/csu/final_csu_file_snomed", 'r') as f:
        for line in f:
            total_text += 1
            if header:
                header = False
                continue
            columns = line.split('\t')
            labels = collapse_label(columns[-1].strip().split('-'))

            text = preprocess_text(columns[4], no_description=True)

            filtered_labels = [l for l in labels if l in snomed_codes_set]

            labels_dist.extend(filtered_labels)

            # we enter another procedure if there's "clinical finding" code in labels
            # if clinical_finding_code in labels:

            spec_labels = columns[-2].strip().split('|')  # specific/raw SNOMED codes

            # Remove any of the leading "S" and "R" in the codes
            # TODO: examine this part very carefully
            # TODO: 'Other disease' + 'Other clinical finding' << 109616
            target_labels = []
            other_disease_code_added = False
            other_clinical_f_added = False
            for s in spec_labels:
                s = s.replace('S', '').replace('R', '')
                if s in other_disese_codes_set:
                    if not other_disease_code_added:
                        target_labels.append('Other_Disease')
                        other_disease_cnt += 1
                        other_disease_code_added = True
                    else:
                        continue

                if s in non_disease_clinical_finding_set:
                    if not other_clinical_f_added:
                        target_labels.append('Other_Clinical_Finding')
                        other_clinical_f_cnt += 1
                        other_clinical_f_added = True
                    else:
                        continue

                if s in lower_snomed_to_higher_map:
                    target_labels.append(lower_snomed_to_higher_map[s])

            # if we did a perfect partition, this should be 0!
            if other_disease_code_added and other_clinical_f_added:
                both_cnt += 1

            seq_labels = collapse_label(target_labels)

            if len(seq_labels) == 0:
                empty_seq_label_cnt += 1

            label_num_per_example.append(len(seq_labels) + len(filtered_labels)) # number of labels per example
            seq_labels_dist.extend(seq_labels)

            # start from 0, and also join back to " " separation
            examples.append([text, " ".join(filtered_labels + seq_labels)])
            # else:
            #     examples.append([text, " ".join(filtered_labels)])

    print "other d codes {}, other clin f codes {}".format(other_disease_cnt, other_clinical_f_cnt)
    print "intersection of other disease codes and non-disease cf codes {}".format(both_cnt)

    # 109616 / 112558 text have clinical finding code
    print "empty seq label text entries are {}".format(empty_seq_label_cnt)

    # import matplotlib.pyplot as plt
    #
    # n, bins, patches = plt.hist(labels_dist, 50, normed=1, facecolor='green', alpha=0.75)
    # plt.show()

    import csv

    # with open("../../data/csu/Files_for_parsing/snomed_ICD_mapped.csv", 'r') as f:
    #     csv_reader = csv.reader(f, delimiter=';')
    #     snomed_code_to_name = {}
    #     for row in csv_reader:
    #         snomed_code_to_name[row[0]] = row[1]

    # print set(seq_labels_dist)
    #
    # seq_labels_dist = count_freq(seq_labels_dist)

    labels_dist = count_freq(labels_dist + seq_labels_dist)

    print labels_dist

    print("number of labels is {} / {}".format(len(labels_dist), len(binned_labels)))

    with open("../../data/csu/Files_for_parsing/refined_snomed_dist.csv", 'wb') as f:
        for k, v in labels_dist.items():
            f.write(binned_labels_names[binned_labels.index(k)] + "," + str(v) + "\n")

    labels_prob = map(lambda t: (t[0], float(t[1]) / sum(labels_dist.values())), labels_dist.items())

    labels_prob = sorted(labels_prob, key=lambda t: t[1])

    print "code, n, p"
    for k, prob in labels_prob:
        print "{}, {}, {}".format(k, labels_dist[k], prob)

    label_list = [t[0] for t in labels_prob]

    # process them into tsv format, but also collect frequency distribution
    serial_numbers = range(len(examples))
    np.random.shuffle(serial_numbers)

    train_numbers = serial_numbers[:int(np.rint(len(examples) * split_proportions['train']))]
    valid_numbers = serial_numbers[
                    int(np.rint(len(examples) * split_proportions['train'])): \
                        int(np.rint(len(examples) * (split_proportions['train'] + split_proportions['valid'])))]
    test_numbers = serial_numbers[
                   int(np.rint(len(examples) * (split_proportions['train'] + split_proportions['valid']))):]

    print(
        "train/valid/test number of examples: {}/{}/{}".format(len(train_numbers), len(valid_numbers),
                                                               len(test_numbers)))

    train, valid, test = [], [], []

    for tn in train_numbers:
        train.append(examples[tn])
    for tn in valid_numbers:
        valid.append(examples[tn])
    for tn in test_numbers:
        test.append(examples[tn])

    write_to_tsv(train, "../../data/csu/snomed_refined_multi_label_no_des_train.tsv", label_list)
    write_to_tsv(valid, "../../data/csu/snomed_refined_multi_label_no_des_valid.tsv", label_list)
    write_to_tsv(test, "../../data/csu/snomed_refined_multi_label_no_des_test.tsv", label_list)

    import json

    with open('../../data/csu/snomed_refined_labels.json', 'wb') as f:
        json.dump(label_list, f)

    names = [binned_labels_names[binned_labels.index(l)] for l in label_list]
    # index matches 0 to 41
    with open('../../data/csu/snomed_refined_labels_to_name.json', 'wb') as f:
        json.dump(names, f)
