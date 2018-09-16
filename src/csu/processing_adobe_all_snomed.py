"""
Process Adobe dataset
and map to all subtypes of disease SNOMED codes
"""

import re
import numpy as np
import json
import pandas as pd
import sys
import copy
import csv
from collections import defaultdict

reload(sys)
sys.setdefaultencoding('utf-8')

vet_tc = pd.read_csv("../../data/csu/Files_for_parsing/snomed_vet_tc.csv", sep='\t')


def subtype_id_test(x):
    temp_df1 = (vet_tc[['subtype']][vet_tc['supertype'] == x])
    results = temp_df1['subtype']
    return results.tolist()


def supertype_id_test(x):
    temp_df1 = (vet_tc[['supertype']][vet_tc['subtype'] == x])
    results = temp_df1['supertype']
    return results.tolist()


def write_to_tsv(data, file_name, label_list):
    # we are translating labels here
    with open(file_name, 'wb') as f:
        for line in data:
            mapped_labels = [str(label_list.index(l)) for l in line[1].split()]
            f.write(line[0].decode('utf-8', 'ignore').encode('utf-8') + '\t' + " ".join(mapped_labels) + '\n')


# the disease level labels we are trying to catch
with open('../../data/csu/snomed_fine_grained_labels.json', 'r') as f:
    snomed_labels = json.load(f)
    snomed_label_set = set(snomed_labels[:-1])
    clinical_finding = '404684003'
    disease_node = '64572001'

with open('../../data/csu/snomed_refined_labels_to_name.json', 'r') as f:
    snomed_names = json.load(f)

np.random.seed(1234)

disease_subtypes = subtype_id_test(int(disease_node))
disease_subtypes = [str(i) for i in disease_subtypes]

def get_most_freq_label(dic):
    most_f_l = None
    most_f_f = 0.
    for l, f in dic.iteritems():
        if f > most_f_f:
            most_f_l = l
    return most_f_l


# this is the correct collapse_label that
# produces "Other Clinical Finding" (that is not disease)
def collapse_label_x(list_labels):
    # collapse and filter at the same time
    flat_list_labels = [item for items in list_labels for item in items if len(item) > 0]
    new_label_set = set()
    for l in flat_list_labels:

        clinical_finding_tag = False
        found_diseas = False
        for super_l in supertype_id_test(int(float(l))):
            # this is strictly searching for disease
            super_l = str(super_l)
            if super_l in snomed_label_set:
                new_label_set.add(super_l)  # set, so ok for repeating add
                found_diseas = True

            if super_l == clinical_finding:
                clinical_finding_tag = True
                # but we only add in the very end

        # if clinical finding tag is present, no disease is, we add it
        if clinical_finding_tag and not found_diseas:
            new_label_set.add(clinical_finding)

    return list(new_label_set)

# this is the one similar to Ashley's generation of CSU dataset
# where disease also counts into clinical finding
def collapse_label(list_labels):
    if clinical_finding not in snomed_label_set:
        snomed_label_set.add(clinical_finding)

    # collapse and filter at the same time
    flat_list_labels = [item for items in list_labels for item in items if len(item) > 0]
    new_label_set = set()
    for l in flat_list_labels:

        for super_l in supertype_id_test(int(float(l))):
            # this is strictly searching for disease
            super_l = str(super_l)
            if super_l in snomed_label_set:
                new_label_set.add(super_l)  # set, so ok for repeating add

    return list(new_label_set)


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = re.sub(r'^https?:\/\/.*[\r\n]*', '', cleantext, flags=re.MULTILINE)
    return cleantext


def clean_date(text):
    cleanr = re.compile('(\d+).(\d+).(\d+)')  # 09/14/16, 9.27.11, 9-27-11
    cleantext = re.sub(cleanr, '', text)
    return cleantext


def remove_date_time(text):
    # starts with more complex pattern, and then match down to simpler patterns
    cleanr = re.compile('((\d+).(\d+).(\d+)) +(\d+).(\d+)[ |pPAa][ |pPAamM].')
    # 09/14/16 1:45 PM JessicaL
    # 09/04/10 3:08pm - katt/Curt
    # 12/13/2013 1101am
    cleantext = re.sub(cleanr, '', text)

    # 7/7/2014 1530
    cleanr = re.compile('((\d+).(\d+).(\d+)) +(\d+).(\d+)')
    cleantext = re.sub(cleanr, '', cleantext)

    # 4/6/16 Nichelason
    cleanr = re.compile('((\d+).(\d+).(\d+)) +')
    cleantext = re.sub(cleanr, '', cleantext)

    # 6/14/11 ak 1359
    # only first part will be removed, not 1359

    return cleantext

import nltk

def preprocess_text(text):
    # this seems to be performing well :)
    no_date_time = remove_date_time(text)

    # simple expansions:
    text = no_date_time.replace('w/', ' with ') # adding space to avoid things like "w/ants"
    text = text.replace('BUT', ' but ') # adding space to avoid things like "w/ants"
    text = text.replace('IS', ' is ') # adding space to avoid things like "w/ants"
    text = text.replace('NOT', ' not ') # adding space to avoid things like "w/ants"
    text = text.replace('(s)', 's')  # eye(s) -> eyes

    matched = 0

    # some abbreviations have '/' like 'V/D' or 'R/O'

    # replace abbr with expanded list
    # first we expand 'OA/AR' into 'OA / AR'

    # text = text.replace('/', ' / ')

    new_text = []
    for word in nltk.word_tokenize(text.decode('latin-1')):
        mt_word = word.strip().lower()
        if mt_word in abbr_dic:
            new_text.append(abbr_dic[mt_word])
            matched += 1
        elif word.isupper() and len(word) >= 4:
            # this will not kill capitalization
            # only longer than 4 characters will trigger such as: "BODY"
            # it's after the abbreviation list
            new_text.append(word.lower())
        elif '/' in mt_word:
            sub_words = mt_word.split('/')
            for sw in sub_words:
                if sw in abbr_dic:
                    new_text.extend([abbr_dic[sw], '/'])
                    matched += 1
                else:
                    new_text.extend([sw, '/'])
            new_text = new_text[:-1]  # get rid of the last extra '/'
        else:
            new_text.append(word.strip())

    one_white_space = ' '.join(new_text)

    return one_white_space, matched


def count_freq(list_labels):
    dic = defaultdict(int)
    for l in list_labels:
        dic[l] += 1
    return dic


if __name__ == '__main__':
    header = True

    abbr_dic = {}

    # vet abbreviations-short_ashley.csv
    with open('../../data/csu/Files_for_parsing/vet abbreviations-short.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for columns in csv_reader:
            exp = re.sub('\(.+\)', '', columns[1])  # get rid of parentheses
            abbr_dic[columns[0].lower()] = exp

    examples = []
    seq_labels_dist = []
    labels_dist = []
    label_num_per_example = []
    total_text = 0

    total_matched = 0

    empty_seq_label_cnt = 0
    file_name = '2018-06-08-Adobe_data_final_snomed_combined.csv'  # Files_for_parsing/adobe_DJ.csv
    # file_name = 'Adobe_data_final_snomed_update_071018.csv'  # Files_for_parsing/adobe_DJ.csv
    with open("../../data/csu/Files_for_parsing/{}".format(file_name), 'r') as f:
        csv_reader = csv.reader(f)
        for columns in csv_reader:
            total_text += 1
            if header:
                header = False
                continue

            # to get rid of rows that don't have labels
            # or have words
            condense_labels = [l.split('|') for l in columns[19:] if len(l) > 0 and not l.lower().islower()]
            if len(condense_labels) == 0:
                continue

            labels = collapse_label(condense_labels)

            text, matched = preprocess_text(columns[9] + ' ' + columns[10] + ' ' + columns[11] + ' ' + columns[12])

            labels_dist.extend(labels)

            examples.append([text, " ".join(labels)])

            total_matched += matched

    labels_dist = count_freq(labels_dist)
    label_list = snomed_labels

    print("number of labels is {}".format(len(labels_dist)))
    print("number of abbreviation matched is {}".format(total_matched))

    print "code, name, n"
    for s_l in snomed_labels:
        print "{}\t{}\t{}".format(s_l, snomed_names[snomed_labels.index(s_l)], labels_dist[s_l])

    # write_to_tsv(examples, "../../data/csu/adobe_snomed_multi_label_no_des_test.tsv", label_list)
    # write_to_tsv(examples, "../../data/csu/adobe_abbr_matched_snomed_multi_label_no_des_test.tsv", label_list)
    # write_to_tsv(examples, "../../data/csu/adobe_combined_abbr_matched_snomed_multi_label_no_des_test.tsv", label_list)
    # write_to_tsv(examples, "../../data/csu/adobe_combined_fixed_abbr_matched_snomed_multi_label_no_des_test.tsv", label_list)
    write_to_tsv(examples, "../../data/csu/adobe_combined_abbr_matched_snomed_fine_grained_label_no_des_test.tsv", label_list)