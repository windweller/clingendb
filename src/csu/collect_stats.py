import csv
import pandas as pd
from collections import defaultdict
import numpy as np
import json
import time
import multiprocessing as mp

vet_tc = pd.read_csv("../../data/csu/Files_for_parsing/snomed_vet_tc.csv", sep='\t')

snomed_id_to_name = {}
snomed_id_index = {}  # can be converted to index
with open("../../data/csu/Files_for_parsing/combined_desc.txt", 'r') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for i, columns in enumerate(csv_reader):
        if i == 0:
            continue
        snomed_id_to_name[columns[4]] = columns[7]  # languageCode:term
        snomed_id_index[columns[4]] = i-1

with open('../../data/csu/snomed_labels.json', 'r') as f:
    snomed_labels = json.load(f)
    snomed_label_set = set(snomed_labels[:-1])
    clinical_finding = '404684003'

with open('../../data/csu/snomed_labels_to_name.json', 'r') as f:
    snomed_names = json.load(f)

def subtype_id_test(x):
    temp_df1 = (vet_tc[['subtype']][vet_tc['supertype'] == x])
    results = temp_df1['subtype']
    return results.tolist()


def supertype_id_test(x):
    temp_df1 = (vet_tc[['supertype']][vet_tc['subtype'] == x])
    results = temp_df1['supertype']
    return results.tolist()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_super_lower_tup(lower):
    list_supers = supertype_id_test(int(float(lower)))
    avail_supers = [str(l) for l in list_supers if str(l) in snomed_label_set]
    return avail_supers

# pool = mp.Pool(processes=4)

# this inner loop is time consuming
# we'll speed up the `supertype_id_test` method
def collapse_label(list_labels, stats_dict):

    # collapse and filter at the same time
    # flat_list_labels = [item for items in list_labels for item in items if len(item) > 0]
    # new_label_set = set()

    numeric_labels = [l for l in list_labels if is_number(l)]

    # pool.map()
    # if len(numeric_labels) > 0:
    #     avail_supers = pool.map(get_super_lower_tup, numeric_labels)
    #
    #     for i in xrange(len(numeric_labels)):
    #         lower_name = snomed_id_to_name[numeric_labels[i]]
    #         for s in avail_supers[i]:
    #             super_name = snomed_names[snomed_labels.index(s)]
    #             stats_dict[super_name][lower_name] += 1

    for lower in numeric_labels:
        avail_supers = get_super_lower_tup(lower)
        for s in avail_supers:
            super_name = snomed_names[snomed_labels.index(s)]
            lower_name = snomed_id_to_name[lower]
            stats_dict[super_name][lower_name] += 1

    # for l in flat_list_labels:
    #     if len(l) == 0:
    #         continue
    #     if not is_number(l):
    #         continue
    #
    #     for super_l in supertype_id_test(int(float(l))):
    #         # this is strictly searching for disease
    #         super_l = str(super_l)
    #         if super_l in snomed_label_set:
    #             new_label_set.add(super_l)  # set, so ok for repeating add
    #             super_name = snomed_names[snomed_labels.index(super_l)]
    #             lower_name = snomed_id_to_name[l]
    #             stats_dict[super_name][lower_name] += 1  # super_label: lower_label


file_name = 'final_csu_file2'
stats_dict = defaultdict(lambda: defaultdict(int))

labels_dist = []
header = True

progress = 0.
total = 112558.
start_time = time.time()
with open("../../data/csu/{}".format(file_name), 'r') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    for columns in csv_reader:
        progress += 1
        if header:
            header = False
            continue

        # to get rid of rows that don't have labels
        # or have words
        condense_labels = columns[5].split('|')
        collapse_label(condense_labels, stats_dict)

        if progress % 100 == 0:
            print "time: {}s, progress: {} / {}".format(time.time() - start_time, progress, total)


with open("../../data/csu/binned_dist_stat.json", 'wb') as f:
    json.dump(stats_dict, f)
