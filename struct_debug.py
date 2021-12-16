import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import pickle
from pathlib import Path

from alphafold.model.quat_affine import QuatAffine
from alphafold.model.folding import FoldIteration, InvariantPointAttention, MultiRigidSidechain
from alphafold.model.folding import generate_affines, compute_renamed_ground_truth
from alphafold.model.folding import backbone_loss, sidechain_loss, structural_violation_loss, find_structural_violations, compute_violation_metrics
from alphafold.model import config

from alphafold.model.all_atom import torsion_angles_to_frames, frames_and_literature_positions_to_atom14_pos
from alphafold.model import r3

if __name__=='__main__':
	model_config = config.model_config('model_1')
	global_config = model_config.model.global_config
	global_config.zero_init = False
	global_config.deterministic = True

	rng = jax.random.PRNGKey(42)

	def test_wrapper_affine(filename, _forward_fn, activations, affine_field_name='affine'):
		qa = QuatAffine.from_tensor(activations)
		feat_this = {**feat, affine_field_name:qa}
		
		apply = hk.transform(_forward_fn).apply
		init = hk.transform(_forward_fn).init
		
		params = init(rng, feat_this)
		params = hk.data_structures.to_mutable_dict(params)
		res = apply(params, rng, feat_this)
		
		feat_save = {'activations':activations, **feat}
		res_path = Path('Debug')/(f'{filename}.pkl')
		with open(res_path, 'wb') as f:
			pickle.dump((feat_save, params, res), f)

	batch_size = 16
	tensor = np.random.rand(batch_size, 7)
	scale = np.random.rand(batch_size)
	point = [np.random.rand(batch_size), np.random.rand(batch_size), np.random.rand(batch_size)]


	N = 200
	M = 200
	C = 64
	Cp = 128

	activations = np.random.rand(N, 7)
	feat = {
		'inputs_1d': jax.random.normal(rng, (N, C), dtype=jnp.float32),
		'inputs_2d': jax.random.normal(rng, (N, M, Cp), dtype=jnp.float32),
		'mask': jnp.asarray(jax.random.bernoulli(rng, 0.5, (N, 1)), dtype=np.float32),
		
	}
	conf = model_config.model.heads.structure_module
	# test_wrapper_affine('InvariantPointAttention',
	# 	lambda batch:InvariantPointAttention(conf, global_config, dist_epsilon=1e-8)(**batch),
	# 	activations=activations
	# )
	
	##TODO
	# feat = {
	# 	'activations': {'act': jax.random.normal(rng, (N, 7), dtype=jnp.float32), 'affine': None},
	# 	'sequence_mask': jnp.asarray(jax.random.bernoulli(rng, 0.5, (N, 1)), dtype=np.float32),
	# 	'is_training': False,
	# 	'initial_act': jax.random.normal(rng, (N, 7), dtype=jnp.float32)
	# }
	# test_wrapper_affine('FoldIteration',
	# 	lambda batch:FoldIteration(conf, global_config)(**batch),
	# 	activations=activations, affine_field_name='update_affine'
	# )

	def test_wrapper(filename, fn, *args):
		res = fn(*args)			
		res_path = Path('Debug')/(f'{filename}.pkl')
		with open(res_path, 'wb') as f:
			pickle.dump((args, res), f)
		return res

	batch_size = 16
	activations = np.random.rand(batch_size, 7)
	aatype = np.random.randint(0, 21, size=(batch_size,), dtype=np.int32)
	torsion_angles_sin_cos = np.random.randn(batch_size, 7, 2)
		
	def test_torsion_angles_to_frames(activations, aatype, torsion_angles_sin_cos):
		rigs = r3.rigids_from_quataffine(QuatAffine.from_tensor(activations))
		res = torsion_angles_to_frames(aatype=aatype, backb_to_global=rigs, torsion_angles_sin_cos=torsion_angles_sin_cos)
		res = r3.rigids_to_tensor_flat9(res)
		return res

	def test_frames_and_literature_positions_to_atom14_pos(activations, aatype, torsion_angles_sin_cos):
		rigs = r3.rigids_from_quataffine(QuatAffine.from_tensor(activations))
		all_frames = torsion_angles_to_frames(aatype=aatype, backb_to_global=rigs, torsion_angles_sin_cos=torsion_angles_sin_cos)
		res = frames_and_literature_positions_to_atom14_pos(aatype=aatype, all_frames_to_global=all_frames)
		res = r3.vecs_to_tensor(res)
		return res

	# test_wrapper('test_torsion_angles_to_frames', test_torsion_angles_to_frames, activations, aatype, torsion_angles_sin_cos)
	# test_wrapper('test_frames_and_literature_positions_to_atom14_pos', test_frames_and_literature_positions_to_atom14_pos, 
	# 				activations, aatype, torsion_angles_sin_cos)