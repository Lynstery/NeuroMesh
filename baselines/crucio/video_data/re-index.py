import os
import re

from crucio.autoencoder.util import IMAGE_EXT, check_work_dir

name = 'demo3'
incre = +3

check_work_dir()

framedir = name+'_frame/'
for filename in os.listdir(framedir):
    if IMAGE_EXT == os.path.splitext(filename)[1]:
        basename = re.sub(r'^[^_]*_', '', os.path.splitext(filename)[0])
        number = int(basename)
        number += incre
        number = '{:04d}'.format(number)
        newfilename = name+'_'+number+'.ext'
        os.rename(framedir+filename, framedir+newfilename)
for filename in os.listdir(framedir):
    newfilename = os.path.splitext(filename)[0]+IMAGE_EXT
    os.rename(framedir+filename, framedir+newfilename)
