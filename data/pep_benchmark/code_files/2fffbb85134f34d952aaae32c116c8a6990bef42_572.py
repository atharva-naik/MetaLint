#!/usr/bin/python

import json
import re

lxx_f     = open('../github.eliranwong/LXX-Rahlfs-1935/template/ossp_wordlist_lxx_only.csv')
lemma_f   = open('lemma', 'w')
strongs_f = open('strongs', 'w')

lxx_f.read(2) # skip BOM

lxx_l = lxx_f.readline()
while lxx_l:

  lxx_l = re.sub(',$', '', lxx_l)
  
  no, txt, json_txt= lxx_l.split('\t') 

  noL = 'L' + no.zfill(5) + ':'

  json_obj=json.loads(json_txt)


  if 'strong' in json_obj:
     strongsG = 'G' + json_obj['strong'].zfill(4)
  else:
     strongsG = ''

  lemma_f  .write(noL + json_obj['lemma'] + '\n')
  strongs_f.write(noL + strongsG          + '\n')

  lxx_l = lxx_f.readline()

lxx_f    .close()
lemma_f  .close()
strongs_f.close()
