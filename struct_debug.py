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

if __name__=='__main__':
	model_config = config.model_config('model_1')
	global_config = model_config.model.global_config
	global_config.zero_init = False
	global_config.deterministic = True

	rng = jax.random.PRNGKey(42)

	def test_wrapper_affine(filename, _forward_fn, activations):
		qa = QuatAffine.from_tensor(activations)
		feat_this = {**feat, 'affine':qa}
		
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
	test_wrapper_affine('InvariantPointAttention',
		lambda batch:InvariantPointAttention(conf, global_config, dist_epsilon=1e-8)(**batch),
		activations=activations
	)