"""
Loads in available associations
and download papers using the FTP services
"""

import time
import os
import requests
import urllib
import json
import logging
import re
from os.path import join as pjoin
from sets import Set

import xml.etree.ElementTree as ElementTree
from xml.etree.ElementTree import tostring
from HTMLParser import HTMLParser  # only works in Python 2.6 - 2.7

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)

h = HTMLParser()

root_folder = './papers'
download_files = False

def get_text(entry):
    # returns None if non-existent, otherwise return text
    if entry is not None:
        return entry.text
    else:
        return None


def remove_xml_tag(text):
    text = re.sub('<[^<]+>', "", text)
    return text


def trim_white_space(text):
    text = re.sub(' +', ' ', text)
    return text


def _save_oa_xml(pubmed_id, pmc_id, download):
    # this saves all XML from efetch
    if download:
        pmc_id_num = int(pmc_id)
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=%d' \
              % pmc_id_num
        response = requests.get(url)
        if response.status_code == requests.codes.ok:
            xml_target = pjoin(root_folder, str(pubmed_id) + '.xml')
        if not os.path.isfile(xml_target):
            with open(xml_target, 'w') as f:
                f.write(response.content)
            return str(xml_target)
        else:
            return None
    else:
        xml_target = pjoin(root_folder, str(pubmed_id) + '.xml')
        return str(xml_target)


def _get_abstract(pubmed_id, pmc_id, download):
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=%d' % int(pmc_id)
    if download:
        response = requests.get(url)
        root = ElementTree.fromstring(response.content)
        # TODO: this leads to a problem
        abstracts = root.find('article/front/article-meta/abstract')  # this leads to a problem
        if abstracts == None:
            return None
        else:
            text = trim_white_space(h.unescape((remove_xml_tag(tostring(abstracts).replace('\n', '')).strip())))
            return text
    else:
        xml_target = pjoin(root_folder, str(pubmed_id) + '.xml')
        root = ElementTree.parse(xml_target)
        abstracts = root.findall('article/front/article-meta/abstract')
        abstract_list = []
        for abstract in abstracts:
            text = trim_white_space(h.unescape((remove_xml_tag(tostring(abstract).replace('\n', '')).strip())))
            abstract_list.append(text)
        text = " ".join(abstract_list)
        return text

def _extract_title(pubmed_id, pmc_id, download):
    if not download:
        xml_target = pjoin(root_folder, str(pubmed_id) + '.xml')
        root = ElementTree.parse(xml_target)
        title = root.find('article/front/article-meta/title-group/article-title')
        if title is not None:
            return trim_white_space(h.unescape((remove_xml_tag(tostring(title).replace('\n', '')).strip())))
        else:
            return None
    else:
        raise NotImplementedError


def download_oa(pubmed_id, pmc_id, wait=0.5, download=True):
    # adapted from Volodymyr's code but made much simpler
    # return abstract text, and address for XML location
    # they get appended to available_associations

    abstract = _get_abstract(pubmed_id, pmc_id, download)
    saved_loc = _save_oa_xml(pubmed_id, pmc_id, download)
    title = _extract_title(pubmed_id, pmc_id, download)

    # to prevent PubMed, we manually throttle
    time.sleep(wait)
    return abstract, saved_loc, title


if __name__ == '__main__':

    with open('oa_simple_allele_associations.json', 'rb') as f:
        available_associations = json.load(f)

    with open('pmdid_to_pmcid.json', 'rb') as f:
        pmdid_to_pmcid = json.load(f)

    logger.info("start downloading")

    # now this downloading is complete, because it won't download
    # repeating papers at all! And create data redundancy
    processed_papers = {}  # PubMed_ID
    counter = 0
    for a in available_associations:
        for i in a['Interpretations']:
            for c in i['Citations']:
                c['PMC_ID'] = pmdid_to_pmcid[c['ID']]
                # and download the article!
                if c['ID'] not in processed_papers:
                    counter += 1
                    abstract, saved_loc, title = download_oa(c['ID'], c['PMC_ID'], download=download_files)
                    processed_papers[c['ID']] = (abstract, saved_loc, title)
                    c['Abstract'] = abstract
                    c['File_Loc'] = saved_loc
                    c['Title'] = title
                else:
                    # this will leave reptitive copies around
                    abstract, saved_loc, title = processed_papers[c['ID']]
                    c['Abstract'] = abstract
                    c['File_Loc'] = saved_loc
                    c['Title'] = title
        if 'ObservedData' in a:
            for ob in a['ObservedData']:
                for c in ob['Citations']:
                    c['PMC_ID'] = pmdid_to_pmcid[c['ID']]
                    if c['ID'] not in processed_papers:
                        counter += 1
                        abstract, saved_loc, title = download_oa(c['ID'], c['PMC_ID'], download=download_files)
                        processed_papers[c['ID']] = (abstract, saved_loc, title)
                    else:
                        abstract, saved_loc, title = processed_papers[c['ID']]
                        c['Abstract'] = abstract
                        c['File_Loc'] = saved_loc
                        c['Title'] = title

        if counter % 100 == 0:
            print "downloaded {} papers".format(counter)

    # save available_associations
    # each citation will have a saved location and an abstract associated with it
    # a bit reptitive.. (we choose redundancy to make our lives easier)
    with open('oa_simple_allele_associations_downloaded.json', 'wb') as f:
        json.dump(available_associations, f)

    # so that we can check the results
    import IPython; IPython.embed()