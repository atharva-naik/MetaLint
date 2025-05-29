#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
Split files to multiple small ones when writing large files.
Usage: write a new class inherits the `MultiFileWriter` class,
       invoke get_file_path to get file path to write to.
"""

import os
import re


class MultiFileWriter(object):

    def __init__(self, max_size=524288000):
        """
        Note: the unit of the argument max_size is byte.
        """
        self.max_size = max_size

    def _get_current(self, base_dir, filename):
        pattern = re.compile(r"^%s-(\d*)" % filename)
        existing_idx_list = []
        for name in os.listdir(base_dir):
            matched = pattern.match(name)
            if matched:
                idx = matched.groups()[0]
                existing_idx_list.append(int(idx))
        if existing_idx_list:
            current_idx = max(existing_idx_list)
            current_file = os.path.join(base_dir,
                                        "%s-%s" % (filename, current_idx))
        else:
            current_idx = 0
            current_file = os.path.join(base_dir, filename)
        return current_idx, current_file

    def get_file_path(self, file_path):
        if not os.path.exists(file_path):
            return file_path

        base_dir = os.path.dirname(file_path)
        filename = os.path.basename(file_path)

        current_idx, current_file = self._get_current(base_dir, filename)
        if os.stat(current_file).st_size < self.max_size:
            return current_file
        return "%s-%s" % (file_path, current_idx + 1)
