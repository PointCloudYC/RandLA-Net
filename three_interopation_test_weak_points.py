import sys, os
import tensorflow as tf
"""
open eager execution for debugging; need next to the 'import tensorflow ...' line, otherwise possibly report ValueError: tf.enable_eager_execution must be called at program startup
COMMENT THIS LINE WHEN TRAINING/EVALUATING
"""
tf.enable_eager_execution()

# custom tf ops based on PointNet++(https://github.com/charlesq34/pointnet2)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/3d_interpolation'))
from tf_interpolate import three_nn, three_interpolate


def three_nearest_interpolation(xyz_query, xyz_support, features_support, batch_inds, k_interpolation=3):
	"""need custom CUDA ops to support (three_nn(), three_interpolate())
	----------
	weakly_points : torch.Tensor
		(n, 3) tensor of the xyz positions of the unknown points
	xyz_support : torch.Tensor
		(B, m, 3) tensor of the xyz positions of the known points (i.e. B PC examples, each is mx3 shape)
	features_support : torch.Tensor
		(B, m, C) tensor of features to be propagated (i.e. B PC examples, each is mx3 shape)
	batch_inds: torch.Tensor
		(n,) tensor of the batch indices to denote which batch for the weakly_points, values are 0 to B-1
	k_interpolation:
		the number of neighbors used for interpolation

	Returns
	-------
	new_features : torch.Tensor
		(n, C2, 1) tensor of the features of the weakly points' features(i.e., n weakly points' new features)
	"""

	# HACK: query features for each weak pt，here treat unknow（n,3) tensor as n sets(i.e., n batches where each batch has 1 pt) such that the MaskedUpSampled code can be used.
	xyz_query = tf.reshape(xyz_query,(tf.shape(xyz_query)[0],1,-1)) # (n,1,3)

	# points_current = points_current[batch_inds,...]  # BUG: CUDA error: an illegal memory access was encountered when use a tensor
	xyz_support = tf.gather(xyz_support, batch_inds, axis=0) # (B,m,3) --> (n,m,3) as each weak point might come from different batch

	# features_current = features_current[batch_inds,...] 
	features_support = tf.gather(features_support,batch_inds,axis=0) # (B,m,C) --> (n,m,C), e.g., (n, 10240,32)

	if xyz_support is not None:
		# query nearest 3 neighbors for each weak point
		dist, idx = three_nn(xyz_query, xyz_support) # (n,1,3), (n,1,3)
		dist_recip = 1.0 / (dist + 1e-8) # (n,1,3)
		norm = tf.reduce_sum(dist_recip, axis=2, keepdims=True) # (n,1,1)
		weight = dist_recip / norm # (n,1,3)

		interpolated_feats = three_interpolate(features_support, idx, weight) # (n,1,C)
	else:
		raise ValueError('make sure the known parameters are valid')

	return interpolated_feats # (n,1,C)

if __name__=='__main__':

    batch_size = 2
    num_points = 40960
    num_weak_points = 6
    num_channels = 32

    # weak points (2,3)
    xyz_query = tf.random.uniform(shape=[num_weak_points,3])

    # support points (Batch=2, 40960, 3)
    xyz_support = tf.random.uniform(shape=[batch_size, num_points, 3]) 

    # support features (Batch=2, 40960, 32)
    feature_support = tf.random.uniform(shape=[batch_size, num_points, num_channels]) 

    # the batch indices for each weak point
    batch_inds = tf.constant([0,0,1,1,1,1])

    interpolated_feats = three_nearest_interpolation(xyz_query, xyz_support, feature_support, batch_inds) # (num_weak_points, num_channels)

    print(interpolated_feats.shape)
    print(interpolated_feats)