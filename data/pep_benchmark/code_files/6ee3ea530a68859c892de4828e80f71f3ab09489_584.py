# -*- coding: utf-8 -*-
import requests
from pyld import jsonld
import json

# Our endpoints
INVENTARIS = 'https://inventaris.onroerenderfgoed.be'
AFBEELDINGEN = 'https://beeldbank.onroerenderfgoed.be/images'
ERFGOEDOBJECTEN = INVENTARIS + '/erfgoedobjecten'
AANDUIDINGSOBJECTEN = INVENTARIS + '/aanduidingsobjecten'
THEMAS = INVENTARIS + '/themas'

def get_data(url, parameters):
    '''
    Fetch all data from a url until there are no more `next` urls in the Link
    header.

    :param str url: The url to fetch from
    :param dict parameters: A dict of query string parameters
    :rtype: dict
    '''
    data = []

    headers = {'Accept': 'application/json'}

    res = requests.get(url, params=parameters, headers=headers)

    data.extend(res.json())

    while 'next' in res.links:
        res = requests.get(res.links['next']['url'], headers=headers)
        data.extend(res.json())

    return data

def add_type(collection, rtype):
    """
    Add the resource type to a resource

    :param list collection: Collection of resources to add a type to
    :param str rtype: The type of all resources in this collection
    :rtype: list
    """
    for c in collection:
        c.update({'@type': rtype})

def add_locatie_samenvatting(afbeeldingen):
    """
    Summarize the location of an image

    :param list afbeeldingen: Collection of afbeeldingen to summarize
    :rtype: list
    """
    for a in afbeeldingen:
        s = ''
        hnr = a.get('location', {}).get('housenumber', {}).get('name')
        straat = a.get('location', {}).get('street', {}).get('name')
        gemeente = a.get('location', {}).get('municipality', {}).get('name')
        prov = a.get('location', {}).get('province', {}).get('name')
        if straat and hnr:
            s = '{} {} ({})'.format(straat, hnr, gemeente)
        elif straat:
            s = '{} ({})'.format(straat, gemeente)
        else:
            s = '{} ({})'.format(gemeente, prov)
        a.update({'locatie_samenvatting': s})

# Determine the CRAB ID for the gemeente you want
# https://loc.geopunt.be/v4/Location?q=knokke-heist
MUNICIPALITY_ID = 191

# Fetch all data
afbeeldingen = get_data(AFBEELDINGEN, {'municipality': MUNICIPALITY_ID})
erfgoedobjecten = get_data(ERFGOEDOBJECTEN, {'gemeente': MUNICIPALITY_ID})
aanduidingsobjecten = get_data(AANDUIDINGSOBJECTEN, {'gemeente': MUNICIPALITY_ID})
themas = get_data(THEMAS, {'gemeente': MUNICIPALITY_ID})

# Add everything together and transform to linked data
inventaris_context = {
    "dct": "http://purl.org/dc/terms/",
    "naam": "dct:title",
    "korte_beschrijving": "dct:description",
    "locatie_samenvatting": "dct:spatial",
    "uri": "@id",
    "Thema": "https://id.erfgoed.net/vocab/ontology#Thema",
    "Erfgoedobject": "https://id.erfgoed.net/vocab/ontology#Erfgoedobject",
    "Aanduidingsobject": "https://id.erfgoed.net/vocab/ontology#Aanduidingsobject"
}
beeldbank_context = {
    "dct": "http://purl.org/dc/terms/",
    "title": "dct:title",
    "description": "dct:description",
    "locatie_samenvatting": "dct:spatial",
    "uri": "@id",
    "Afbeelding": "https://purl.org/dc/dcmiType/Image"
}

# Add types to all datasets and location summary to images
add_type(erfgoedobjecten, "Erfgoedobject")
erfgoedobjecten = jsonld.expand(erfgoedobjecten, {'expandContext':inventaris_context})
add_type(aanduidingsobjecten, "Aanduidingsobject")
aanduidingsobjecten = jsonld.expand(aanduidingsobjecten, {'expandContext':inventaris_context})
add_type(themas, "Thema")
themas = jsonld.expand(themas, {'expandContext':inventaris_context})
add_type(afbeeldingen, "Afbeelding")
add_locatie_samenvatting(afbeeldingen)
afbeeldingen = jsonld.expand(afbeeldingen, {'expandContext':beeldbank_context})

# Add all datasets together
stuff = erfgoedobjecten + aanduidingsobjecten + themas + afbeeldingen

# Compact all data to simplify the keys we're working with
dct_context = {
    "dct": "http://purl.org/dc/terms/",
    "title": "dct:title",
    "description": "dct:description",
    "spatial": "dct:spatial",
    "uri": "@id",
    "type": "@type",
    "Thema": "https://id.erfgoed.net/vocab/ontology#Thema",
    "Erfgoedobject": "https://id.erfgoed.net/vocab/ontology#Erfgoedobject",
    "Aanduidingsobject": "https://id.erfgoed.net/vocab/ontology#Aanduidingsobject",
    "Afbeelding": "https://purl.org/dc/dcmiType/Image"
}
compactstuff = jsonld.compact(stuff, dct_context)

# Print all records to the screen
for s in compactstuff['@graph']:
    h = '{}'.format(s['title'])
    print(h)
    print(len(h)*'=')
    print('Type: {}'.format(s['type']))
    print('URI: {}'.format(s['uri']))
    print('Location: {}'.format(s['spatial']))
    if 'description' in s and s['description']:
        print(s['description'])
    print()