"""
This is to generate fine-grained label training dataset
"""

import re
import numpy as np
import pandas as pd
import json

"""
We train without diagnosis, and with multilabel
"""

vet_tc = pd.read_csv("../../data/csu/Files_for_parsing/snomed_vet_tc.csv", sep='\t')

snomed_map = pd.read_csv('./Files_for_parsing/snomed_ICD_mapped_no17.csv')


def supertype_id(x):
    temp_df1 = (vet_tc[['supertype']][vet_tc['subtype'] == x])
    temp_df2 = pd.merge(temp_df1, snomed_map, on="supertype")
    if ~temp_df2.empty:
        result = temp_df2['supertype'].values
        return (result)


def subtype_id_test(x):
    temp_df1 = (vet_tc[['subtype']][vet_tc['supertype'] == x])
    results = temp_df1['subtype']
    return results.tolist()


def supertype_id_test(x):
    temp_df1 = (vet_tc[['supertype']][vet_tc['subtype'] == x])
    results = temp_df1['supertype']
    return results.tolist()


np.random.seed(1234)

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
    dic = {}
    for l in list_labels:
        if l not in dic:
            dic[l] = 1
        else:
            dic[l] += 1
    return dic


def get_most_freq_label(dic):
    most_f_l = None
    most_f_f = 0.
    for l, f in dic.iteritems():
        if f > most_f_f:
            most_f_l = l
    return most_f_l


# def collapse_label_x(list_labels):
#     # collapse and filter at the same time
#     flat_list_labels = list_labels.split('|')
#     new_label_set = set()
#     for l in flat_list_labels:
#
#         # both string match
#         if l not in snomed_code_to_name:
#             continue
#
#         # clinical_finding_tag = False
#         # found_diseas = False
#
#         supertypes = set(supertype_id_test(int(float(l))))
#         matched_types = supertypes.intersection(subtypes_of_disease_code)
#
#         new_label_set = new_label_set.union(matched_types)
#
#     return [str(c) for c in new_label_set]

def collapse_label_x(list_labels):
    flat_list_labels = list_labels.split('|')
    new_label_set = set()
    for l in flat_list_labels:
        if l not in snomed_code_to_name:
            continue
        found_super_types = supertype_id(int(l))
        new_label_set = new_label_set.union(found_super_types)

    return [str(c) for c in new_label_set]


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = re.sub(r'^https?:\/\/.*[\r\n]*', '', cleantext, flags=re.MULTILINE)
    return cleantext


# TODO: 2. Preserve things like "Texas A&M", the ampersand in the middle
def preprocess_text(text, no_description):
    no_html = cleanhtml(text)
    one_white_space = ' '.join(no_html.split())
    no_html_entities = re.sub('&[a-z]+;', '', one_white_space)

    if no_description:
        # delete both diagnosis and discharge status
        no_html_entities = no_html_entities.split('Diagnosis:')[0]

    return no_html_entities


with open('../../data/csu/snomed_labels.json', 'r') as f:
    snomed_labels = json.load(f)
    snomed_label_set = set(snomed_labels[:-1])
    clinical_finding = '404684003'

with open('../../data/csu/snomed_labels_to_name.json', 'r') as f:
    snomed_names = json.load(f)

if __name__ == '__main__':

    import csv

    with open("../../data/csu/Files_for_parsing/combined_desc.txt", 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        snomed_code_to_name = {}
        for row in csv_reader:
            snomed_code_to_name[row[4]] = row[7]

    header = True

    # subtypes_of_disease_code = set(subtype_id_test(int('64572001')))

    examples = []
    labels_dist = []
    with open("../../data/csu/final_csu_file2", 'r') as f:
        for num, line in enumerate(f):
            if header:
                header = False
                continue
            columns = line.split('\t')
            labels = columns[-2]

            text = preprocess_text(columns[4], no_description=True)

            seq_labels = collapse_label_x(labels)
            labels_dist.extend(seq_labels)
            # start from 0, and also join back to " " separation
            examples.append([text, "|".join(seq_labels)])

            if num % 1000 == 0:
                print(num)

    # import matplotlib.pyplot as plt
    #
    # n, bins, patches = plt.hist(labels_dist, 50, normed=1, facecolor='green', alpha=0.75)
    # plt.show()

    labels_dist = count_freq(labels_dist)

    print("number of labels is {}".format(len(labels_dist)))

    labels_prob = map(lambda t: (t[0], float(t[1]) / sum(labels_dist.values())), labels_dist.items())

    labels_prob = sorted(labels_prob, key=lambda t: t[1])

    snomed_code_to_prob = {}

    print "code, n, p"
    for k, prob in labels_prob:
        print "{}, {}, {}".format(snomed_code_to_name[k], labels_dist[k], prob)
        snomed_code_to_prob[k] = prob

    with open("../../data/csu/snomed_disease_codes_dist.csv", 'wb') as f:
        csv_writer = csv.writer(f)
        for k, v in labels_dist.items():
            csv_writer.writerow([snomed_code_to_name[k], str(v), str(snomed_code_to_prob[k])])
            # f.write(snomed_code_to_name[k] + "," + str(v) + "," + str(snomed_code_to_prob[k]) + "\n")

    label_list = [t[0] for t in labels_prob]

    import csv

    with open('../../data/csu/final_csu_file_disease', 'w') as f:
        csv_writer = csv.writer(f)
        for ex in examples:
            csv_writer.writerow(ex)

            # process them into tsv format, but also collect frequency distribution
            # serial_numbers = range(len(examples))
            # np.random.shuffle(serial_numbers)
            #
            # train_numbers = serial_numbers[:int(np.rint(len(examples) * split_proportions['train']))]
            # valid_numbers = serial_numbers[
            #                 int(np.rint(len(examples) * split_proportions['train'])): \
            #                     int(np.rint(len(examples) * (split_proportions['train'] + split_proportions['valid'])))]
            # test_numbers = serial_numbers[
            #                int(np.rint(len(examples) * (split_proportions['train'] + split_proportions['valid']))):]
            #
            # print(
            #     "train/valid/test number of examples: {}/{}/{}".format(len(train_numbers), len(valid_numbers),
            #                                                            len(test_numbers)))
            #
            # train, valid, test = [], [], []
            #
            # for tn in train_numbers:
            #     train.append(examples[tn])
            # for tn in valid_numbers:
            #     valid.append(examples[tn])
            # for tn in test_numbers:
            #     test.append(examples[tn])
            #
            # write_to_tsv(train, "../../data/csu/snomed_fine_grained_multi_label_no_des_train.tsv", label_list)
            # write_to_tsv(valid, "../../data/csu/snomed_fine_grained_multi_label_no_des_valid.tsv", label_list)
            # write_to_tsv(test, "../../data/csu/snomed_fine_grained_multi_label_no_des_test.tsv", label_list)
            #
            # import json
            # with open('../../data/csu/snomed_fine_grained_labels.json', 'wb') as f:
            #     json.dump(label_list, f)
            #
            # names = [snomed_code_to_name[l] for l in label_list]
            # # index matches 0 to 41
            # with open('../../data/csu/snomed_fine_grained_labels_to_name.json', 'wb') as f:
            #     json.dump(names, f)
