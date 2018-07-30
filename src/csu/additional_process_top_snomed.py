"""
Process top-level SNOMED code

This one already assumes some base files that exist
"""

"""
Takes in raw data and preprocess this into good format
for TorchText to train on
"""

import re
import numpy as np

"""
We train without diagnosis, and with multilabel
"""

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

def collapse_label(labels):
    # Note: for SNOMED we no longer take out category 17 (no longer exist)
    labels = labels.strip()
    # labels = labels.replace('17', '')
    list_labels = filter(lambda l: len(l) > 0, labels.split('-'))
    set_labels = set(list_labels)  # remove redundancies
    return list(set_labels)

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = re.sub(r'^https?:\/\/.*[\r\n]*', '', cleantext, flags=re.MULTILINE)
    return cleantext

# TODO: 2. Preserve things like "Texas A&M", the ampersand in the middle
def preprocess_text(text):
    no_html = cleanhtml(text)
    one_white_space = ' '.join(no_html.split())
    no_html_entities = re.sub('&[a-z]+;', '', one_white_space)

    return no_html_entities

def extract_examples(filename, header=True):
    examples = []
    labels_dist = []
    with open("../../data/csu/revised_CSU_input_files/{}".format(filename), 'r') as f:
        for line in f:
            if header:
                header = False
                continue
            columns = line.split('\t')
            case_id = columns[0]

            text = preprocess_text(columns[2])  # "Description field is taken care of by Ashley"

            seq_labels = collapse_label(case_id_to_label_map[case_id])
            labels_dist.extend(seq_labels)
            # start from 0, and also join back to " " separation
            examples.append([text, " ".join(seq_labels)])

    return examples, labels_dist

def examples_to_csv(examples, csv_file_prefix):
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

    write_to_tsv(train, "../../data/csu/snomed_{}_multi_label_no_des_train.tsv".format(csv_file_prefix), label_list)
    write_to_tsv(valid, "../../data/csu/snomed_{}_multi_label_no_des_valid.tsv".format(csv_file_prefix), label_list)
    write_to_tsv(test, "../../data/csu/snomed_{}_multi_label_no_des_test.tsv".format(csv_file_prefix), label_list)


if __name__ == '__main__':
    header = True

    case_id_to_label_map = {}
    with open("../../data/csu/final_csu_file2", 'r') as f:
        for line in f:
            if header:
                header = False
                continue
            columns = line.split('\t')
            if columns[0] in case_id_to_label_map:
                raise Exception("{} already exists in the map".format(columns[0]))
            case_id_to_label_map[columns[0]] = columns[-1]

    medsum_all_fields_no_diagnosis_examples, _ = extract_examples('medsum_all_fields_no_diagnosis')
    medsum_revised_fields1_no_diagnosis_examples, _ = extract_examples('medsum_revised_fields1_no_diagnosis')

    # import matplotlib.pyplot as plt
    #
    # n, bins, patches = plt.hist(labels_dist, 50, normed=1, facecolor='green', alpha=0.75)
    # plt.show()

    import json
    with open('../../data/csu/snomed_labels.json', 'rb') as f:
        label_list = json.load(f)  # an actual list!

    examples_to_csv(medsum_all_fields_no_diagnosis_examples, 'all_fields')
    examples_to_csv(medsum_revised_fields1_no_diagnosis_examples, 'revised_fields')