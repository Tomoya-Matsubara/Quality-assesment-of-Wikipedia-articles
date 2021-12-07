# -*- coding: utf-8 -*-

'''
pip install py-readability-metrics
'''


import re, os, csv, time
from readability import Readability
# Uncomment these in the first run:
#import nltk
#nltk.download('punkt')

class Features:
  def __init__(self, text=None):
    self.text = text
      
  def structural_features(self):
    '''
    Returns a list of structural features
    -------
    length
        the length of the whole text (markup included)
    references
        the number of references found with "<ref" tag
    links
        the number of links leading to other Wikipedia pages (images and categories included)
    citations_template
        the number of citations utilizing a standard template (e.g. book, article etc. - https://en.wikipedia.org/wiki/Wikipedia:Citation_templates)
    citations_non_template
        the number of citations which are not utilizing a standard template
    categories
        the number of categories referred to the article
    infobox
        indicate the presence of the infobox
    images
        the number of images in the article
    l2_headings
        the number of level 2 headings (bigger)
    l3plus_headings
        the number of level 3 (or more) headings (smaller)
    languages
        the number of languages of the article
    '''
    length = len(self.text.encode("utf8"))
    references = self.text.count("<ref")
    links = len(re.findall("(\[\[(.*?)\]\])", self.text))
    citations_template = self.text.count("{{cite") + self.text.count("{{Citation")
    citations_non_template = references - citations_template
    categories = self.text.count("[[Category:")
    infobox = self.text.count("{{Infobox")
    images = self.text.count("[[Image:")
    l2_headings = len(re.findall("\n(==)\w+", self.text))
    l3plus_headings = len(re.findall("\n(===+)\w+", self.text))
    languages = len(re.findall("(\[\[([a-z]{2}:))", self.text))
    return [length, references, links, citations_template, citations_non_template, categories, infobox, images, l2_headings, l3plus_headings, languages]

  def readability_scores(self):
    '''
    Returns a list of readability scores
    -------
    Flesch Reading Ease
    Flesch-Kincaid Grade Level
    SMOG
    Coleman Liau Index
    Automated Readability Index (ARI)
    Dale Chall Readability
    Linsear Write
    Gunning Fog
    SPACHE
    '''
    # Remove all the content between tags [], <>, {}
    text_body = re.sub("[\{\[\<].*?[\>\}\]]", "", self.text)
    # Remove all the tag referring to headings
    text_body = re.sub("==+", "", text_body)
    
    r = Readability(text_body)
    scores = []
    readability_functions = [r.flesch, r.flesch_kincaid, r.smog, r.coleman_liau, r.ari, r.dale_chall, r.linsear_write, r.gunning_fog, r.spache]
    for function in readability_functions:
        try:
            scores.append(function().score)
        except:
            scores.append(float('inf'))
    #grades = [[r.flesch().grade_levels], r.flesch_kincaid().grade_level, r.smog().grade_level, r.coleman_liau().grade_level, [r.ari().grade_levels], [r.dale_chall().grade_levels], r.linsear_write().grade_level, r.gunning_fog().grade_level, ]
    #print(grades)
    return [round(score, 2) for score in scores]  
    
  def getFeatures(self, text=""):
    # Return the concatenation of the structural features and the readability scores
    self.text = text
    return self.structural_features() + self.readability_scores()


start_time = time.time()
p = Features()
n = 0
with open("features.csv", 'w', newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["length", "references", "links", "citations_template", "citations_non_template", "categories", "infobox", "images", "l2_headings", "l3plus_headings", "languages", "flesch", "flesch_kincaid", "smog", "coleman_liau", "ari", "dale_chall", "linsear_write", "gunning_fog", "spache"])
    with os.scandir("./text") as directory:
        for path in directory:
            n += 1
            file = open(path, "r")
            data = file.read()
            if(data != ""):
                writer.writerow(p.getFeatures(data))
            else:
                raise ValueError("Article " + str(n) + " is empty")
        
print("Extracted features of " + str(n) + " articles in " + str(time.time() - start_time) + " seconds")

