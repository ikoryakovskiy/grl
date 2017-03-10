"""RL data container."""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse

sys.path.append('/home/ivan/work/scripts/py')
from my_csv.utils import *

hd = """COLUMNS:
time[0], 
state0[0], 
state0[1], 
state0[2], 
state0[3], 
state0[4], 
state0[5], 
state0[6], 
state0[7], 
state0[8], 
state0[9], 
state0[10], 
state0[11], 
state0[12], 
state0[13], 
state0[14], 
state0[15]{contact}
DATA:"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-c', '--add_contact_info', help='Adding contact information', action='store_true')
    args = parser.parse_args()

    hd_sz = get_header_size(args.filename)
    print "Skipping {} rows".format(hd_sz)

    data = np.loadtxt(args.filename, skiprows=hd_sz, delimiter=',')
    ts = data[:, 0]       # time
    xs = data[:, 1:17]    # states
    xc = data[:, -1]      # contacts

    # join
    if args.add_contact_info:
        data = np.column_stack((ts,xs,xc))
        out_header = hd.format(contact=",\ncontact")
    else:
        data = np.column_stack((ts,xs))
        out_header = hd.format(contact="")

    # save
    [path, fn_ext] = os.path.split(args.filename)
    if path == "":
        path = "."
    [fn, ext] = os.path.splitext(fn_ext)
    pt = path+"/"+fn+"-converted"+ext
    print "Saving to {}".format(pt)
    np.savetxt(pt, data, fmt='%11.6f', delimiter=',', newline='\n', header=out_header, comments='')

if __name__ == "__main__":
    main()



