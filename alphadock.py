import os
import sys
import pathlib
from absl import logging
from typing import Dict
import argparse
import random
import time
import subprocess
import pickle
import numpy as np
import json
import re
from tqdm import tqdm

import jax
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.relax import relax
from alphafold.data import parsers
from alphafold.data.tools import jackhmmer

import collabfold as cf

from matplotlib import gridspec
import matplotlib.pyplot as plt

TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

def run_jackhmmer(sequence, prefix, args):	
	fasta_path = f"{prefix}.fasta"
	with open(fasta_path, 'wt') as f:
		f.write(f'>query\n{sequence}')

	pickled_msa_path = f"{prefix}.jackhmmer.pickle"
	if os.path.isfile(pickled_msa_path):
		msas_dict = pickle.load(open(pickled_msa_path,"rb"))
		msas, deletion_matrices, names = (msas_dict[k] for k in ['msas', 'deletion_matrices', 'names'])
		full_msa = []
		for msa in msas:
			full_msa += msa
	else:
		dbs = []

		num_jackhmmer_chunks = {'uniref90': 59, 'smallbfd': 17, 'mgnify': 71}
		total_jackhmmer_chunks = sum(num_jackhmmer_chunks.values())
		with tqdm(total=total_jackhmmer_chunks, bar_format=TQDM_BAR_FORMAT) as pbar:
			def jackhmmer_chunk_callback(i):
				pbar.update(n=1)

			pbar.set_description('Searching uniref90')
			jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
					binary_path=args.jackhmmer_binary_path,
					database_path=args.uniref90_database_path,
					get_tblout=True,
					num_streamed_chunks=None,
					streaming_callback=None,
					z_value=135301051)
			dbs.append(('uniref90', jackhmmer_uniref90_runner.query(fasta_path)))

			pbar.set_description('Searching smallbfd')
			jackhmmer_smallbfd_runner = jackhmmer.Jackhmmer(
					binary_path=args.jackhmmer_binary_path,
					database_path=args.small_bfd_database_path,
					get_tblout=True,
					num_streamed_chunks=None,
					streaming_callback=None,
					z_value=65984053)
			dbs.append(('smallbfd', jackhmmer_smallbfd_runner.query(fasta_path)))

			pbar.set_description('Searching mgnify')
			jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
					binary_path=args.jackhmmer_binary_path,
					database_path=args.mgnify_database_path,
					get_tblout=True,
					num_streamed_chunks=None,
					streaming_callback=None,
					z_value=304820129)
			dbs.append(('mgnify', jackhmmer_mgnify_runner.query(fasta_path)))

		# --- Extract the MSAs and visualize ---
		# Extract the MSAs from the Stockholm files.
		# NB: deduplication happens later in pipeline.make_msa_features.

		mgnify_max_hits = 501
		msas = []
		deletion_matrices = []
		names = []
		for db_name, db_results in dbs:
			unsorted_results = []
			for i, result in enumerate(db_results):
				msa, deletion_matrix, target_names = parsers.parse_stockholm(result['sto'])
				e_values_dict = parsers.parse_e_values_from_tblout(result['tbl'])
				e_values = [e_values_dict[t.split('/')[0]] for t in target_names]
				zipped_results = zip(msa, deletion_matrix, target_names, e_values)
				if i != 0:
					# Only take query from the first chunk
					zipped_results = [x for x in zipped_results if x[2] != 'query']
				unsorted_results.extend(zipped_results)
			sorted_by_evalue = sorted(unsorted_results, key=lambda x: x[3])
			db_msas, db_deletion_matrices, db_names, _ = zip(*sorted_by_evalue)
			if db_msas:
				if db_name == 'mgnify':
					db_msas = db_msas[:mgnify_max_hits]
					db_deletion_matrices = db_deletion_matrices[:mgnify_max_hits]
					db_names = db_names[:mgnify_max_hits]
				msas.append(db_msas)
				deletion_matrices.append(db_deletion_matrices)
				names.append(db_names)
				msa_size = len(set(db_msas))
				print(f'{msa_size} Sequences Found in {db_name}')

			pickle.dump({"msas":msas, "deletion_matrices":deletion_matrices, "names":names}, open(pickled_msa_path,"wb"))
	return msas, deletion_matrices, names

def _check_flag(flag: str, preset: str, should_be_set: bool):
	if should_be_set != bool(flag):
		verb = 'be' if should_be_set else 'not be'
		raise ValueError(f'{flag} must {verb} set for preset "{preset}"')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-jobname', default='debug', type=str)
	parser.add_argument('-sequence', default='PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK:AISRSLDAPLTSVRVIITEMAKGHFGIGGELASK', type=str)
	parser.add_argument('-homooligomer', default='1', type=str)
	
	parser.add_argument('-output_dir', default='/media/HDD/AlphaFold2Output', type=str)
	parser.add_argument('-data_dir', default='/media/HDD/AlphaFold2', type=str)
	
	parser.add_argument('-jackhmmer_binary_path', default='/usr/bin/jackhmmer', type=str)
	parser.add_argument('-hhblits_binary_path', default='/usr/bin/hhblits', type=str)
	parser.add_argument('-hhsearch_binary_path', default='/usr/bin/hhsearch', type=str)
	parser.add_argument('-kalign_binary_path', default='/usr/bin/kalign', type=str)

	parser.add_argument('-uniref90_database_path', default='uniref90/uniref90.fasta', type=str)
	parser.add_argument('-mgnify_database_path', default='mgnify/mgy_clusters_2018_12.fa', type=str)
	parser.add_argument('-bfd_database_path', default='bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt', type=str)
	parser.add_argument('-small_bfd_database_path', default='small_bfd/bfd-first_non_consensus_sequences.fasta', type=str)
	parser.add_argument('-uniclust30_database_path', default='uniclust30/uniclust30_2018_08/uniclust30_2018_08', type=str)
	parser.add_argument('-pdb70_database_path', default='pdb70/pdb70', type=str)
	parser.add_argument('-template_mmcif_dir', default='pdb_mmcif/mmcif_files', type=str)
	parser.add_argument('-obsolete_pdbs_path', default='pdb_mmcif/obsolete.dat', type=str)
	
	parser.add_argument('-max_template_date', default='2020-05-14', type=str)
	parser.add_argument('-preset', default='reduced_dbs', type=str)
	parser.add_argument('-benchmark', default=False, type=int)
	parser.add_argument('-random_seed', default=None, type=int)
	
	args = parser.parse_args()
	args.uniref90_database_path = os.path.join(args.data_dir, args.uniref90_database_path)
	args.mgnify_database_path = os.path.join(args.data_dir, args.mgnify_database_path)
	args.bfd_database_path = os.path.join(args.data_dir, args.bfd_database_path)
	args.small_bfd_database_path = os.path.join(args.data_dir, args.small_bfd_database_path)
	args.uniclust30_database_path = os.path.join(args.data_dir, args.uniclust30_database_path)
	args.pdb70_database_path = os.path.join(args.data_dir, args.pdb70_database_path)
	args.template_mmcif_dir = os.path.join(args.data_dir, args.template_mmcif_dir)
	args.obsolete_pdbs_path = os.path.join(args.data_dir, args.obsolete_pdbs_path)

	args.jackhmmer_binary_path = subprocess.run(["which", "jackhmmer"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8')
	args.hhblits_binary_path = subprocess.run(["which", "hhblits"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8')
	args.hhsearch_binary_path = subprocess.run(["which", "hhsearch"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8')
	args.kalign_binary_path = subprocess.run(["which", "kalign"], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8')
	
	MAX_TEMPLATE_HITS = 20
	RELAX_MAX_ITERATIONS = 0
	RELAX_ENERGY_TOLERANCE = 2.39
	RELAX_STIFFNESS = 10.0
	RELAX_EXCLUDE_RESIDUES = []
	RELAX_MAX_OUTER_ITERATIONS = 20

	use_small_bfd = args.preset == 'reduced_dbs'
	if use_small_bfd:
		args.bfd_database_path = None
		args.uniclust30_database_path = None
	_check_flag(args.small_bfd_database_path, args.preset, should_be_set=use_small_bfd)
	_check_flag(args.bfd_database_path, args.preset, should_be_set=not use_small_bfd)
	_check_flag(args.uniclust30_database_path, args.preset, should_be_set=not use_small_bfd)

	
	# define sequence
	sequence = re.sub("[^A-Z:/]", "", args.sequence.upper())
	sequence = re.sub(":+",":",sequence)
	sequence = re.sub("/+","/",sequence)

	jobname = re.sub(r'\W+', '', args.jobname)

	# define number of copies
	homooligomer = re.sub("[:/]+",":", args.homooligomer)
	if len(homooligomer) == 0: homooligomer = "1"
	homooligomer = re.sub("[^0-9:]", "", homooligomer)
	homooligomers = [int(h) for h in homooligomer.split(":")]

	ori_sequence = sequence
	sequence = sequence.replace("/","").replace(":","")
	seqs = ori_sequence.replace("/","").split(":")

	if len(seqs) != len(homooligomers):
		if len(homooligomers) == 1:
			homooligomers = [homooligomers[0]] * len(seqs)
			homooligomer = ":".join([str(h) for h in homooligomers])
		else:
			while len(seqs) > len(homooligomers):
				homooligomers.append(1)
			homooligomers = homooligomers[:len(seqs)]
			homooligomer = ":".join([str(h) for h in homooligomers])
			print("WARNING: Mismatch between number of breaks ':' in 'sequence' and 'homooligomer' definition")

	full_sequence = "".join([s*h for s,h in zip(seqs,homooligomers)])

	# prediction directory
	output_dir = os.path.join(args.output_dir, jobname)
	os.makedirs(output_dir, exist_ok=True)
	# delete existing files in working directory
	for f in os.listdir(output_dir):
		os.remove(os.path.join(output_dir, f))

	MIN_SEQUENCE_LENGTH = 16
	MAX_SEQUENCE_LENGTH = 2500

	aatypes = set('ACDEFGHIKLMNPQRSTVWY')  # 20 standard aatypes
	if not set(full_sequence).issubset(aatypes):
		raise Exception(f'Input sequence contains non-amino acid letters: {set(sequence) - aatypes}. AlphaFold only supports 20 standard amino acids as inputs.')
	if len(full_sequence) < MIN_SEQUENCE_LENGTH:
		raise Exception(f'Input sequence is too short: {len(full_sequence)} amino acids, while the minimum is {MIN_SEQUENCE_LENGTH}')
	if len(full_sequence) > MAX_SEQUENCE_LENGTH:
		raise Exception(f'Input sequence is too long: {len(full_sequence)} amino acids, while the maximum is {MAX_SEQUENCE_LENGTH}. Please use the full AlphaFold system for long sequences.')

	if len(full_sequence) > 1400:
		print(f"WARNING: For a typical Google-Colab-GPU (16G) session, the max total length is ~1400 residues. You are at {len(full_sequence)}! Run Alphafold may crash.")

	print(f"homooligomer: '{homooligomer}'")
	print(f"total_length: '{len(full_sequence)}'")
	print(f"working_directory: '{output_dir}'")

	msa_method = "jackhmmer"
	pair_msa = False #@param {type:"boolean"}
	pair_cov = 50 #@param [50,75,90] {type:"raw"}
	pair_qid = 20 #@param [15,20,30,40,50] {type:"raw"}
	include_unpaired_msa = True #@param {type:"boolean"}
	add_custom_msa = False #@param {type:"boolean"}
	msa_format = "fas"
	os.makedirs('tmp', exist_ok=True)
	msas, deletion_matrices = [],[]

	seqs = ori_sequence.replace('/','').split(':')
	_blank_seq = ["-" * len(seq) for seq in seqs]
	_blank_mtx = [[0] * len(seq) for seq in seqs]
	def _pad(ns,vals,mode):
		if mode == "seq": _blank = _blank_seq.copy()
		if mode == "mtx": _blank = _blank_mtx.copy()
		if isinstance(ns, list):
			for n,val in zip(ns,vals): _blank[n] = val
		else: _blank[ns] = vals
		if mode == "seq": return "".join(_blank)
		if mode == "mtx": return sum(_blank,[])

	if not pair_msa or (pair_msa and include_unpaired_msa):
		# gather msas
		if msa_method == "mmseqs2":
			prefix = cf.get_hash("".join(seqs))
			prefix = os.path.join('tmp',prefix)
			print(f"running mmseqs2")
			A3M_LINES = cf.run_mmseqs2(seqs, prefix, filter=True)

		for n, seq in enumerate(seqs):
			# tmp directory
			prefix = cf.get_hash(seq)
			prefix = os.path.join('tmp',prefix)

			if msa_method == "mmseqs2":
				# run mmseqs2
				a3m_lines = A3M_LINES[n]
				msa, mtx = parsers.parse_a3m(a3m_lines)
				msas_, mtxs_ = [msa],[mtx]

			elif msa_method == "jackhmmer":
				print(f"running jackhmmer on seq_{n}")
				# run jackhmmer
				msas_, mtxs_, names_ = ([sum(x,())] for x in run_jackhmmer(seq, prefix, args))
			
			# pad sequences
			for msa_,mtx_ in zip(msas_,mtxs_):
				msa,mtx = [sequence],[[0]*len(sequence)]      
				for s,m in zip(msa_,mtx_):
					msa.append(_pad(n,s,"seq"))
					mtx.append(_pad(n,m,"mtx"))

				msas.append(msa)
				deletion_matrices.append(mtx)

	# save MSA as pickle
	pickle.dump({"msas":msas,"deletion_matrices":deletion_matrices},
							open(os.path.join(output_dir,"msa.pickle"),"wb"))

#########################################
# Merge and filter
#########################################
	msa_merged = sum(msas,[])
	if len(msa_merged) > 1:
		print(f'{len(msa_merged)} Sequences Found in Total')

	ok = dict.fromkeys(range(len(msa_merged)),True)

	Ln = np.cumsum(np.append(0,[len(seq) for seq in seqs]))
	Nn,lines = [],[]
	n,new_msas,new_mtxs = 0,[],[]
	for msa,mtx in zip(msas,deletion_matrices):
		new_msa,new_mtx = [],[]
		for s,m in zip(msa,mtx):
			if n in ok:
				new_msa.append(s)
				new_mtx.append(m)
			n += 1
		if len(new_msa) > 0:
			new_msas.append(new_msa)
			new_mtxs.append(new_mtx)
			Nn.append(len(new_msa))
			msa_ = np.asarray([list(seq) for seq in new_msa])
			gap_ = msa_ != "-"
			qid_ = msa_ == np.array(list(sequence))
			gapid = np.stack([gap_[:,Ln[i]:Ln[i+1]].max(-1) for i in range(len(seqs))],-1)
			seqid = np.stack([qid_[:,Ln[i]:Ln[i+1]].mean(-1) for i in range(len(seqs))],-1).sum(-1) / gapid.sum(-1)
			non_gaps = gap_.astype(np.float)
			non_gaps[non_gaps == 0] = np.nan
			lines.append(non_gaps[seqid.argsort()]*seqid[seqid.argsort(),None])

	msas = new_msas
	deletion_matrices = new_mtxs

	Nn = np.cumsum(np.append(0,Nn))

#########################################
# Display
#########################################

	lines = np.concatenate(lines,0)
	if len(lines) > 1:
		plt.figure(figsize=(8,5),dpi=100)
		plt.title("Sequence coverage")
		plt.imshow(	lines,
					interpolation='nearest', aspect='auto',
					cmap="rainbow_r", vmin=0, vmax=1, origin='lower',
					extent=(0, lines.shape[1], 0, lines.shape[0]))
		for i in Ln[1:-1]:
			plt.plot([i,i],[0,lines.shape[0]],color="black")
		
		for j in Nn[1:-1]:
			plt.plot([0,lines.shape[1]],[j,j],color="black")

		plt.plot((np.isnan(lines) == False).sum(0), color='black')
		plt.xlim(0,lines.shape[1])
		plt.ylim(0,lines.shape[0])
		plt.colorbar(label="Sequence identity to query",)
		plt.xlabel("Positions")
		plt.ylabel("Sequences")
		plt.savefig(os.path.join(output_dir,"msa_coverage.png"), bbox_inches = 'tight', dpi=200)
		# plt.show()

	

	num_relax = "None"
	rank_by = "pLDDT" #@param ["pLDDT","pTMscore"]
	use_turbo = True #@param {type:"boolean"}
	max_msa = "512:1024" #@param ["512:1024", "256:512", "128:256", "64:128", "32:64"]
	max_msa_clusters, max_extra_msa = [int(x) for x in max_msa.split(":")]
	show_images = False
	num_models = 5 #@param [1,2,3,4,5] {type:"raw"}
	use_ptm = True #@param {type:"boolean"}
	num_ensemble = 1 #@param [1,8] {type:"raw"}
	max_recycles = 3 #@param [1,3,6,12,24,48] {type:"raw"}
	tol = 0 #@param [0,0.1,0.5,1] {type:"raw"}
	is_training = False #@param {type:"boolean"}
	num_samples = 1 #@param [1,2,4,8,16,32] {type:"raw"}
	subsample_msa = True #@param {type:"boolean"}

	save_pae_json = True 
	save_tmp_pdb = True
	if use_ptm == False and rank_by == "pTMscore":
		print("WARNING: models will be ranked by pLDDT, 'use_ptm' is needed to compute pTMscore")
		rank_by = "pLDDT"
	
#############################
# delete old files
#############################
	for f in os.listdir(output_dir):
		if "rank_" in f:
			os.remove(os.path.join(output_dir, f))

#############################
# homooligomerize
#############################
	lengths = [len(seq) for seq in seqs]
	msas_mod, deletion_matrices_mod = cf.homooligomerize_heterooligomer(msas, deletion_matrices, lengths, homooligomers)

#############################
# define input features
#############################
	def _placeholder_template_feats(num_templates_, num_res_):
		return {
				'template_aatype': np.zeros([num_templates_, num_res_, 22], np.float32),
				'template_all_atom_masks': np.zeros([num_templates_, num_res_, 37, 3], np.float32),
				'template_all_atom_positions': np.zeros([num_templates_, num_res_, 37], np.float32),
				'template_domain_names': np.zeros([num_templates_], np.float32),
				'template_sum_probs': np.zeros([num_templates_], np.float32),
		}

	num_res = len(full_sequence)
	feature_dict = {}
	feature_dict.update(pipeline.make_sequence_features(full_sequence, 'test', num_res))
	feature_dict.update(pipeline.make_msa_features(msas_mod, deletion_matrices=deletion_matrices_mod))
	if not use_turbo:
		feature_dict.update(_placeholder_template_feats(0, num_res))

	def do_subsample_msa(F, N=10000, random_seed=0):
		'''subsample msa to avoid running out of memory'''
		M = len(F["msa"])
		if N is not None and M > N:
			print(f"whhhaaa... too many sequences ({M}) subsampling to {N}")
			np.random.seed(random_seed)
			idx = np.append(0,np.random.permutation(np.arange(1,M)))[:N]
			F_ = {}
			F_["msa"] = F["msa"][idx]
			F_["deletion_matrix_int"] = F["deletion_matrix_int"][idx]
			F_["num_alignments"] = np.full_like(F["num_alignments"],N)
			for k in ['aatype', 'between_segment_residues',
								'domain_name', 'residue_index',
								'seq_length', 'sequence']:
								F_[k] = F[k]
			return F_
		else:
			return F

################################
# set chain breaks
################################
	Ls = []
	for seq,h in zip(ori_sequence.split(":"),homooligomers):
		Ls += [len(s) for s in seq.split("/")] * h

	Ls_plot = sum([[len(seq)]*h for seq,h in zip(seqs,homooligomers)],[])

	feature_dict['residue_index'] = cf.chain_break(feature_dict['residue_index'], Ls)

###########################
# run alphafold
###########################
	def parse_results(prediction_result, processed_feature_dict):
		b_factors = prediction_result['plddt'][:,None] * prediction_result['structure_module']['final_atom_mask']  
		dist_bins = jax.numpy.append(0,prediction_result["distogram"]["bin_edges"])
		dist_mtx = dist_bins[prediction_result["distogram"]["logits"].argmax(-1)]
		contact_mtx = jax.nn.softmax(prediction_result["distogram"]["logits"])[:,:,dist_bins < 8].sum(-1)

		out = {"unrelaxed_protein": protein.from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors),
					"plddt": prediction_result['plddt'],
					"pLDDT": prediction_result['plddt'].mean(),
					"dists": dist_mtx,
					"adj": contact_mtx}
		if "ptm" in prediction_result:
			out.update({"pae": prediction_result['predicted_aligned_error'],
									"pTMscore": prediction_result['ptm']})
		return out

	model_names = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5'][:num_models]
	total = len(model_names) * num_samples
	with tqdm(total=total, bar_format=TQDM_BAR_FORMAT) as pbar:
		#######################################################################
		# precompile model and recompile only if length changes
		#######################################################################
		if use_turbo:
			name = "model_5_ptm" if use_ptm else "model_5"
			N = len(feature_dict["msa"])
			L = len(feature_dict["residue_index"])
			compiled = (N, L, use_ptm, max_recycles, tol, num_ensemble, max_msa, is_training)
			if "COMPILED" in dir():
				if COMPILED != compiled: recompile = True
			else: recompile = True
			if recompile:
				cf.clear_mem("gpu")
				cfg = config.model_config(name)      

				# set size of msa (to reduce memory requirements)
				msa_clusters = min(N, max_msa_clusters)
				cfg.data.eval.max_msa_clusters = msa_clusters
				cfg.data.common.max_extra_msa = max(min(N-msa_clusters,max_extra_msa),1)

				cfg.data.common.num_recycle = max_recycles
				cfg.model.num_recycle = max_recycles
				cfg.model.recycle_tol = tol
				cfg.data.eval.num_ensemble = num_ensemble

				params = data.get_model_haiku_params(name, args.data_dir)
				model_runner = model.RunModel(cfg, params, is_training=is_training)
				COMPILED = compiled
				recompile = False

		else:
			cf.clear_mem("gpu")
			recompile = True

		# cleanup
		if "outs" in dir(): del outs
		outs = {}
		cf.clear_mem("cpu")  

		#######################################################################
		for num, model_name in enumerate(model_names): # for each model
			name = model_name+"_ptm" if use_ptm else model_name

			# setup model and/or params
			params = data.get_model_haiku_params(name, args.data_dir)
			if use_turbo:
				for k in model_runner.params.keys():
					model_runner.params[k] = params[k]
			else:
				cfg = config.model_config(name)
				cfg.data.common.num_recycle = cfg.model.num_recycle = max_recycles
				cfg.model.recycle_tol = tol
				cfg.data.eval.num_ensemble = num_ensemble
				model_runner = model.RunModel(cfg, params, is_training=is_training)

			for seed in range(num_samples): # for each seed
				# predict
				key = f"{name}_seed_{seed}"
				pbar.set_description(f'Running {key}')
				if subsample_msa:
					subsampled_N = int(3E7/L)
					sampled_feats_dict = do_subsample_msa(feature_dict, N=subsampled_N, random_seed=seed)    
					processed_feature_dict = model_runner.process_features(sampled_feats_dict, random_seed=seed)
				else:
					processed_feature_dict = model_runner.process_features(feature_dict, random_seed=seed)

				prediction_result, (r, t) = cf.to(model_runner.predict(processed_feature_dict, random_seed=seed),"cpu")
				outs[key] = parse_results(prediction_result, processed_feature_dict)
				
				# report
				pbar.update(n=1)
				line = f"{key} recycles:{r} tol:{t:.2f} pLDDT:{outs[key]['pLDDT']:.2f}"
				if use_ptm: line += f" pTMscore:{outs[key]['pTMscore']:.2f}"
				print(line)
				if show_images:
					fig = cf.plot_protein(outs[key]["unrelaxed_protein"], Ls=Ls_plot, dpi=100)
					plt.show()
				if save_tmp_pdb:
					tmp_pdb_path = os.path.join(output_dir,f'unranked_{key}_unrelaxed.pdb')
					pdb_lines = protein.to_pdb(outs[key]["unrelaxed_protein"])
					with open(tmp_pdb_path, 'w') as f: f.write(pdb_lines)


				# cleanup
				del processed_feature_dict, prediction_result
				if subsample_msa: del sampled_feats_dict

			if use_turbo:
				del params
			else:
				del params, model_runner, cfg
				cf.clear_mem("gpu")

		# delete old files
		for f in os.listdir(output_dir):
			if "rank" in f:
				os.remove(os.path.join(output_dir, f))

		# Find the best model according to the mean pLDDT.
		model_rank = list(outs.keys())
		model_rank = [model_rank[i] for i in np.argsort([outs[x][rank_by] for x in model_rank])[::-1]]

		# Write out the prediction
		for n,key in enumerate(model_rank):
			prefix = f"rank_{n+1}_{key}" 
			pred_output_path = os.path.join(output_dir,f'{prefix}_unrelaxed.pdb')
			
			pdb_lines = protein.to_pdb(outs[key]["unrelaxed_protein"])
			with open(pred_output_path, 'w') as f:
				f.write(pdb_lines)
				
	############################################################
	print(f"model rank based on {rank_by}")
	for n,key in enumerate(model_rank):
		print(f"rank_{n+1}_{key} {rank_by}:{outs[key][rank_by]:.2f}")
		if use_ptm and save_pae_json:
			pae = outs[key]["pae"]
			max_pae = pae.max()
			# Save pLDDT and predicted aligned error (if it exists)
			pae_output_path = os.path.join(output_dir,f'rank_{n+1}_{key}_pae.json')
			# Save predicted aligned error in the same format as the AF EMBL DB
			rounded_errors = np.round(np.asarray(pae), decimals=1)
			indices = np.indices((len(rounded_errors), len(rounded_errors))) + 1
			indices_1 = indices[0].flatten().tolist()
			indices_2 = indices[1].flatten().tolist()
			pae_data = json.dumps([{'residue1': indices_1,
									'residue2': indices_2,
									'distance': rounded_errors.flatten().tolist(),
									'max_predicted_aligned_error': max_pae.item()
							}],
							indent=None,
							separators=(',', ':'))

			with open(pae_output_path, 'w') as f:
				f.write(pae_data)
	

	