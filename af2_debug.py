import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import pickle
from pathlib import Path

from alphafold.model.modules import Attention, GlobalAttention, MSARowAttentionWithPairBias, MSAColumnAttention, MSAColumnGlobalAttention
from alphafold.model.modules import TriangleAttention, TriangleMultiplication, OuterProductMean
from alphafold.model import config

def test_wrapper(filename, _forward_fn):
	apply = hk.transform(_forward_fn).apply
	init = hk.transform(_forward_fn).init
	
	params = init(rng, feat)
	params = hk.data_structures.to_mutable_dict(params)
	
	res = apply(params, rng, feat)

	res_path = Path('Debug')/(f'{filename}.pkl')
	with open(res_path, 'wb') as f:
		pickle.dump((feat, params, res), f)
	

if __name__=='__main__':
	model_config = config.model_config('model_1')
	model_config.model.global_config.zero_init = False
	rng = jax.random.PRNGKey(42)
	
	batch_size = 4
	N_queries = 64
	N_keys = 64
	q_channels = 32
	m_channels = 32

	N_seq = 64
	N_res = 128
	c_z = 16
	c_m = 32


	
	feat = {
	# 'q_data': jnp.zeros((batch_size, N_queries, q_channels), dtype=jnp.float32),
	# 'm_data': jnp.zeros((batch_size, N_keys, m_channels), dtype=jnp.float32),
	# 'bias': jnp.zeros((batch_size, 1, N_queries, N_keys), dtype=jnp.float32),
	# 'nonbatched_bias': jnp.zeros((N_queries, N_keys), dtype=jnp.float32),
	'q_data': jax.random.normal(rng, (batch_size, N_queries, q_channels), dtype=jnp.float32),
	'm_data': jax.random.normal(rng, (batch_size, N_keys, m_channels), dtype=jnp.float32),
	'bias': jax.random.normal(rng, (batch_size, 1, N_queries, N_keys), dtype=jnp.float32),
	'nonbatched_bias': jax.random.normal(rng, (N_queries, N_keys), dtype=jnp.float32),
	'q_mask': jax.random.bernoulli(rng, 0.5, (batch_size, N_queries, q_channels)),

	'msa_act': jax.random.normal(rng, (N_seq, N_res, c_m), dtype=jnp.float32),
	'msa_mask': jax.random.bernoulli(rng, 0.5, (N_seq, N_res)),
	'pair_act': jax.random.normal(rng, (N_res, N_res, c_z), dtype=jnp.float32),
	'pair_mask': jax.random.bernoulli(rng, 0.5, (N_res, N_res)),
	}

	conf = model_config.model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias
	global_config = model_config.model.global_config
	# conf.gating = False
	# test_wrapper('Attention',
	# 	lambda batch:
	# 		Attention(conf, global_config, output_dim=256)
	# 		(q_data=batch['q_data'], m_data=batch['m_data'], bias=batch['bias'], nonbatched_bias=batch['nonbatched_bias'])
	# 	)

	# test_wrapper('GlobalAttention',
	# 	lambda batch:
	# 		GlobalAttention(conf, global_config, output_dim=256)
	# 		(q_data=batch['q_data'], m_data=batch['m_data'], q_mask=batch['q_mask'], bias=batch['bias'])
	# 	)

	# test_wrapper('MSARowAttentionWithPairBias',
	# 	lambda batch:
	# 		MSARowAttentionWithPairBias(conf, global_config)
	# 		(msa_act=batch['msa_act'], msa_mask=batch['msa_mask'], pair_act=batch['pair_act'], is_training=False)
	# 	)

	# conf = model_config.model.embeddings_and_evoformer.evoformer.msa_column_attention
	# test_wrapper('MSAColumnAttention',
	# 	lambda batch:
	# 		MSAColumnAttention(conf, global_config)
	# 		(msa_act=batch['msa_act'], msa_mask=batch['msa_mask'], is_training=False)
	# 	)
	
	# test_wrapper('MSAColumnGlobalAttention',
	# 	lambda batch:
	# 		MSAColumnGlobalAttention(conf, global_config)
	# 		(msa_act=batch['msa_act'], msa_mask=batch['msa_mask'], is_training=False)
	# 	)

	
	# conf = model_config.model.embeddings_and_evoformer.evoformer.triangle_attention_starting_node
	# test_wrapper('TriangleAttention',
	# 	lambda batch:
	# 		TriangleAttention(conf, global_config)
	# 		(pair_act=batch['pair_act'], pair_mask=batch['pair_mask'], is_training=False)
	# 	)
	# conf = model_config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing
	# test_wrapper('TriangleMultiplication',
	# 	lambda batch:
	# 		TriangleMultiplication(conf, global_config)
	# 		(act=batch['pair_act'], mask=batch['pair_mask'], is_training=False)
	# 	)
	# conf = model_config.model.embeddings_and_evoformer.evoformer.outer_product_mean
	# test_wrapper('OuterProductMean',
	# 	lambda batch:
	# 		OuterProductMean(conf, global_config, num_output_channel=256)
	# 		(act=batch['msa_act'], mask=batch['msa_mask'], is_training=False)
	# 	)
	
	
	

	