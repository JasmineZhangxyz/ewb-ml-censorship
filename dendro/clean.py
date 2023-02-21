#!/usr/bin/env python3

import errno
import os
import shutil
import sys

def clean(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.pyc'):
                path = os.path.join(root, file)
                print(path)
                os.remove(path)
        try:
            dirs.remove('__pycache__')
        except ValueError:
            pass
        else:
            path = os.path.join(root, '__pycache__')
            print(path)
            shutil.rmtree(path)
    try:
        shutil.rmtree(os.path.join(dir, 'build'))
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

if __name__ == '__main__':
    clean(os.path.dirname(os.path.join('.', sys.argv[0])))
