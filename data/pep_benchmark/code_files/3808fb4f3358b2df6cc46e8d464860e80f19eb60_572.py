import re
from datetime import datetime
from dateutil import parser as dt_parser, relativedelta as dt_relativedelta
import logging

log = logging.getLogger(__name__)

def dot_split(s):
    return [part for part in re.split(r"(?<!\.)\.(?!\.)", s)]


def split_strip(_str, on=',', remove_empty=True):
    lst = (_str if isinstance(_str, list) else _str.split(on))
    lst = [e.strip() for e in lst]
    if remove_empty:
        lst = list(filter(bool, lst))
    return lst


def str2rdt(strdt):
    matches = dict(
        s = 'seconds',
        m = 'minutes',
        h = 'hours',
        d = 'days',
        M = 'months',
        y = 'years'
    )

    # is it a relative date ?
    rg = re.compile('(([-+]?)(\d+))([smhdMy])\\b', re.DOTALL)
    m = rg.search(strdt)
    if m:
        number = int(m.group(1))
        word = m.group(4)
        if word in matches:
            return dt_relativedelta.relativedelta(**{matches[word]:number})


def str2dt(strdt, _raise=False):
    if not strdt:
        raise ValueError('Datetime string can not be empty or None')

    if isinstance(strdt, datetime):
        return strdt

    dt = str2rdt(strdt)
    if dt:
        return datetime.utcnow()+dt

    try:
        return dt_parser.parse(strdt)
    except ValueError as e:
        msg = 'Datetime string `%s` not recognized as datetime. Did you miss +- signs for relative dates?' % strdt
        if _raise:
            raise ValueError(msg)
        else:
            log.error(msg)
