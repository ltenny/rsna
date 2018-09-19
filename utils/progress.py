from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

class Progress(object):
    @staticmethod
    def show_progress(counter, interval = 10):
        if counter % interval == 0:
            print('    %d' % counter, end='\r')
            sys.stdout.flush()
