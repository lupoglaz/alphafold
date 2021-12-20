import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import pickle
from pathlib import Path

from alphafold.model.quat_affine import QuatAffine

if __name__=='__main__':

	def test_wrapper(filename, fn, *args):
		res = fn(*args)			
		res_path = Path('Debug')/(f'{filename}.pkl')
		with open(res_path, 'wb') as f:
			pickle.dump((args, res), f)
		return res

	batch_size = 16
	tensor = np.random.rand(batch_size, 7)
	scale = np.random.rand(batch_size)
	point = [np.random.rand(batch_size), np.random.rand(batch_size), np.random.rand(batch_size)]
	update = np.random.rand(batch_size, 6)

	def test_init(activations):
		affine = QuatAffine.from_tensor(activations)
		return affine.quaternion, affine.translation, affine.rotation
	
	def test_scale_translation(activations, scale):
		affine = QuatAffine.from_tensor(activations)
		q = affine.scale_translation(scale)
		return q.quaternion, q.translation, q.rotation

	def test_apply_rot_func(activations):
		affine = QuatAffine.from_tensor(activations)
		q = affine.apply_rotation_tensor_fn(lambda t: t+1.0)
		return q.quaternion, q.translation, q.rotation

	def test_to_tensor(activations):
		affine = QuatAffine.from_tensor(activations)
		q = affine.apply_rotation_tensor_fn(lambda t: t+1.0)
		res = q.to_tensor()
		return res

	def test_apply_to_point(activations, point):
		affine = QuatAffine.from_tensor(activations)
		res = affine.apply_to_point(point)
		return res

	def test_invert_point(activations, point):
		affine = QuatAffine.from_tensor(activations)
		res = affine.invert_point(point)
		return res
	
	def test_pre_compose(activations, update):
		affine = QuatAffine.from_tensor(activations)
		res = affine.pre_compose(update)
		res = res.to_tensor()
		return res
	
	# test_wrapper('quat_init', test_init, tensor)
	# test_wrapper('quat_scale_translation', test_scale_translation, tensor, scale)
	# test_wrapper('quat_apply_rotation_func', test_apply_rot_func, tensor)
	# test_wrapper('quat_to_tensor', test_to_tensor, tensor)
	# test_wrapper('quat_apply_to_point', test_apply_to_point, tensor, point)
	# test_wrapper('quat_invert_point', test_invert_point, tensor, point)
	test_wrapper('quat_pre_compose', test_pre_compose, tensor, update)