import os
import math
import re

patch_size = 128

# Bird files
bfiles = os.listdir('bird')

# Complete files
f = open('complete.csv', 'r')
fdata = f.read()
f.close()
cfiles = str.split(fdata, '\n')

# Writing to this file
f = open('mapping.csv', 'w')

from hachoir.parser import createParser
from hachoir.metadata import extractMetadata

f.write('filename,patch_size,num_height,num_width,bird_indicies (space separated)\n')

for i in range(0, len(cfiles)):
	cfile = cfiles[i]
	if cfile is not '':
		base_dir = 'S:\\Thesis_Storage\\GameCamera-2016\\'

		filename = os.path.join(base_dir, cfile)
		parser = createParser(filename)
		metadata = extractMetadata(parser)
		im_width = metadata.get('width')
		im_height = metadata.get('height')
		num_height = math.floor((im_height - patch_size)/patch_size)
		num_width = math.floor(im_width/patch_size)
		match_files = [bfile for bfile in bfiles if re.sub('\\\\', '_', cfile)[:-4] in bfile]
		match_numbers = [re.findall('([0-9]+)\.png$', tstfile)[0] for tstfile in match_files]
		bird_indicies = ' '.join(match_numbers)
		f.write(cfile + ',' + 
			str(patch_size) + ',' +
			str(num_height) + ',' + 
			str(num_width) + ',' + 
			bird_indicies + '\n')

f.close()