import os
import sys
import argparse
from pathlib import Path

def check_complete(args, name):
	output_path = Path(args.output_dir)/Path(name)
	if not output_path.exists():
		return False
		
	num_out = 0
	for path in output_path.iterdir():
		if path.suffix == '.pdb' and (str(path.name)[:4] == 'rank'):
			num_out += 1
	return num_out == 5

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-input_file', default='FoldTargets.dat', type=str)
	parser.add_argument('-output_dir', default='/media/HDD/AlphaFold2Output', type=str)
	args = parser.parse_args()

	with open(args.input_file, 'rt') as fin:
		lines = fin.readlines()
		for i in range(0, len(lines), 2):
			header = lines[i].split('|')
			name = header[0][1:]
			olig = header[1][:-1]
			seq = lines[i+1][:-1]
			print(name)
			print(olig)
			print(seq)
			if check_complete(args, name):
				continue
			os.system(f'python alphadock.py -jobname {name} -sequence {seq} -homooligomer {olig}')