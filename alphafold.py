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


def _check_flag(flag: str, preset: str, should_be_set: bool):
	if should_be_set != bool(flag):
		verb = 'be' if should_be_set else 'not be'
		raise ValueError(f'{flag} must {verb} set for preset "{preset}"')

def predict_structure(	fasta_path: str,
						fasta_name: str,
						output_dir_base: str,
						data_pipeline: pipeline.DataPipeline,
						model_runners: Dict[str, model.RunModel],
						amber_relaxer: relax.AmberRelaxation,
						benchmark: bool,
						random_seed: int):
  
	"""Predicts structure using AlphaFold for the given sequence."""
  
	timings = {}
	output_dir = os.path.join(output_dir_base, fasta_name)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	msa_output_dir = os.path.join(output_dir, 'msas')
	if not os.path.exists(msa_output_dir):
		os.makedirs(msa_output_dir)

	# Get features.
	t_0 = time.time()
	feature_dict = data_pipeline.process(input_fasta_path=fasta_path, msa_output_dir=msa_output_dir)
	timings['features'] = time.time() - t_0

	# Write out features as a pickled dictionary.
	features_output_path = os.path.join(output_dir, 'features.pkl')
	with open(features_output_path, 'wb') as f:
		pickle.dump(feature_dict, f, protocol=4)
	with open(features_output_path, 'rb') as f:
		feature_dict = pickle.load(f)

	relaxed_pdbs = {}
	plddts = {}

	# Run the models.
	for model_name, model_runner in model_runners.items():
		logging.info('Running model %s', model_name)
		t_0 = time.time()
		processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
		timings[f'process_features_{model_name}'] = time.time() - t_0

		proc_features_output_path = os.path.join(output_dir, 'proc_features.pkl')
		with open(proc_features_output_path, 'wb') as f:
			pickle.dump(processed_feature_dict, f, protocol=4)
		with open(proc_features_output_path, 'rb') as f:
			processed_feature_dict = pickle.load(f)

		t_0 = time.time()
		prediction_result, _ = model_runner.predict(processed_feature_dict)
		t_diff = time.time() - t_0
		timings[f'predict_and_compile_{model_name}'] = t_diff
		logging.info('Total JAX model %s predict time (includes compilation time, see --benchmark): %.0f?', model_name, t_diff)

		if benchmark:
			t_0 = time.time()
			model_runner.predict(processed_feature_dict)
			timings[f'predict_benchmark_{model_name}'] = time.time() - t_0

		# Get mean pLDDT confidence metric.
		plddt = prediction_result['plddt']
		plddts[model_name] = np.mean(plddt)

		# Save the model outputs.
		result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
		with open(result_output_path, 'wb') as f:
			pickle.dump(prediction_result, f, protocol=4)

		# Add the predicted LDDT in the b-factor column.
		# Note that higher predicted LDDT value means higher model confidence.
		plddt_b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)
		unrelaxed_protein = protein.from_prediction(
			features=processed_feature_dict,
			result=prediction_result,
			b_factors=plddt_b_factors)

		unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
		with open(unrelaxed_pdb_path, 'w') as f:
			f.write(protein.to_pdb(unrelaxed_protein))

		# Relax the prediction.
		t_0 = time.time()
		relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
		timings[f'relax_{model_name}'] = time.time() - t_0

		relaxed_pdbs[model_name] = relaxed_pdb_str

		# Save the relaxed PDB.
		relaxed_output_path = os.path.join(output_dir, f'relaxed_{model_name}.pdb')
		with open(relaxed_output_path, 'w') as f:
			f.write(relaxed_pdb_str)

	# Rank by pLDDT and write out relaxed PDBs in rank order.
	ranked_order = []
	for idx, (model_name, _) in enumerate(sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
		ranked_order.append(model_name)
		ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
		with open(ranked_output_path, 'w') as f:
			f.write(relaxed_pdbs[model_name])

	ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
	with open(ranking_output_path, 'w') as f:
		f.write(json.dumps({'plddts': plddts, 'order': ranked_order}, indent=4))

	logging.info('Final timings for %s: %s', fasta_name, timings)

	timings_output_path = os.path.join(output_dir, 'timings.json')
	with open(timings_output_path, 'w') as f:
		f.write(json.dumps(timings, indent=4))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-fasta_path', default='T1024.fas', type=str)
	parser.add_argument('-output_dir', default='/media/HDD/AlphaFold2Output', type=str)
	parser.add_argument('-model_name', default='model_1', type=str)
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
	
	template_featurizer = templates.TemplateHitFeaturizer(
		mmcif_dir=args.template_mmcif_dir,
		max_template_date=args.max_template_date,
		max_hits=MAX_TEMPLATE_HITS,
		kalign_binary_path=args.kalign_binary_path,
		release_dates_path=None,
		obsolete_pdbs_path=args.obsolete_pdbs_path)

	data_pipeline = pipeline.DataPipeline(
		jackhmmer_binary_path=args.jackhmmer_binary_path,
		hhblits_binary_path=args.hhblits_binary_path,
		hhsearch_binary_path=args.hhsearch_binary_path,
		uniref90_database_path=args.uniref90_database_path,
		mgnify_database_path=args.mgnify_database_path,
		bfd_database_path=args.bfd_database_path,
		uniclust30_database_path=args.uniclust30_database_path,
		small_bfd_database_path=args.small_bfd_database_path,
		pdb70_database_path=args.pdb70_database_path,
		template_featurizer=template_featurizer,
		use_small_bfd=use_small_bfd)

	model_runners = {}
	model_config = config.model_config(args.model_name)
	model_config.data.eval.num_ensemble = 1
	model_params = data.get_model_haiku_params(model_name=args.model_name, data_dir=args.data_dir)
	model_runner = model.RunModel(model_config, model_params)
	model_runners[args.model_name] = model_runner

	logging.info('Have %d models: %s', len(model_runners), list(model_runners.keys()))

	amber_relaxer = relax.AmberRelaxation(
	  max_iterations=RELAX_MAX_ITERATIONS,
	  tolerance=RELAX_ENERGY_TOLERANCE,
	  stiffness=RELAX_STIFFNESS,
	  exclude_residues=RELAX_EXCLUDE_RESIDUES,
	  max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)

	random_seed = args.random_seed
	if random_seed is None:
		random_seed = random.randrange(sys.maxsize)
	logging.info('Using random seed %d for the data pipeline', random_seed)

	# Predict structure for each of the sequences.
	predict_structure(
		fasta_path=args.fasta_path,
		fasta_name=pathlib.Path(args.fasta_path).stem,
		output_dir_base=args.output_dir,
		data_pipeline=data_pipeline,
		model_runners=model_runners,
		amber_relaxer=amber_relaxer,
		benchmark=args.benchmark,
		random_seed=random_seed)