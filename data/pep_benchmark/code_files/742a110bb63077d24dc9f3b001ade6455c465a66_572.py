import logging, simplejson, pprint, re, os, sys
from urllib2 import urlopen
from urllib import urlencode
from datetime import datetime
from string import Template
from Levenshtein import ratio, median

from pyramid.httpexceptions import HTTPFound
from pyramid_handlers import action

from beaker.cache import CacheManager

import cdp_viz.handlers.base as base
import cdp_viz.models as model
from cdp_viz.lib.timeUtils import getDatetimeFromString, getISODateTimeString
from cdp_viz.lib.sparql import MD5_SPARQL_TMPL, MANIFEST_SPARQL_TMPL, sparqlQuery
from cdp_viz.lib.sessionGraph import rdf2sessionGraph

log = logging.getLogger(__name__)

CDE_PACKAGE_TMPL = Template('''#!/bin/sh
wget "${LDOS_BASE_URL}/data/download?hash=${hash}" -O $file
for i in `cat session_manifest.txt`; do
  file=`echo $$i | awk 'BEGIN{FS=","}{print $$1}'`
  filebase=`basename $$file`
  dir=`dirname $$file`
  md5=`echo $$i | awk 'BEGIN{FS=","}{print $$2}'`
  oct_perms=`echo $$i | awk 'BEGIN{FS=","}{print $$6}'`
  perms=`python -c "print oct(int($$oct_perms))[-3:]"`
  mkdir -p $$dir
  wget -q "${LDOS_BASE_URL}/data/download?file=$${filebase}&hash=$${md5}" -O $$file
  chmod $$perms $$file
  echo "downloaded: $$file"
done
''')

class Download(base.Handler):
    @action(renderer="string")
    def sessionEntities(self):
        sessionId = self.request.params.get('sessionId')
        #log.debug("sessionId: %s" % sessionId)
        d = simplejson.loads(sparqlQuery(MD5_SPARQL_TMPL.substitute(uri=sessionId)))
        #log.debug(pprint.pformat(d))
        wgetLines = []
        for res in d['results']['bindings']:
            entity = res['entity']['value']
            hash = res['md5']['value']
            match = re.search(r'http://provenance\.jpl\.nasa\.gov/cdp#(.*?)/\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}.*?$', entity)
            if match:
                file = os.path.basename(match.group(1))
                wgetLines.append("wget %s/data/download?hash=%s -O %s" % 
                    (self.request.registry.settings['ldos.url'], hash, file))
        return "#!/bin/sh\n%s\n" % "\n".join(wgetLines)

    @action(renderer="string")
    def cde(self):
        self.request.response.content_disposition = 'attachment; filename="wget_cde_package.sh"'
        sessionId = self.request.params.get('sessionId')
        #log.debug("sessionId: %s" % sessionId)
        d = simplejson.loads(sparqlQuery(MANIFEST_SPARQL_TMPL.substitute(uri=sessionId)))
        #log.debug(pprint.pformat(d))
        wgetLines = []
        for res in d['results']['bindings']:
            loc = res['loc']['value']
            hash = res['md5']['value']
            match = re.search(r'(.*?)(?:/\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}.*?)?$', loc)
            if match:
                file = os.path.basename(match.group(1))
                return CDE_PACKAGE_TMPL.substitute(LDOS_BASE_URL=self.request.registry.settings['ldos.url'],
                                                   hash=hash,
                                                   file=file)
        return "No CDE package for session %s." % sessionId

    def download(self):
        filename = self.request.params.get('filename')
        md5 = self.request.params.get('hash')
        return HTTPFound(location="%s/data/download?filename=%s&hash=%s" % (
                         self.request.registry.settings['ldos.url'],
                         filename, md5))
