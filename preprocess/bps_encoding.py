import numpy as np
from sklearn.neighbors import NearestNeighbors


########################### bps encoding  ############################
# Sample uniformly in the unit ball (random)
def bps_gen_ball_inside(n_bps=1000, random_seed=100):
    np.random.seed(random_seed)
    x = np.random.normal(size=[n_bps, 3])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms  # points on the unit ball surface
    r = np.random.uniform(size=[n_bps, 1])
    u = np.power(r, 1.0 / 3)
    basis_set = 1 * x_unit * u  # basic set coordinates, [n_bps, 3]
    return basis_set


# basis_set: unit ball basis points, [n_bps, 3]
# scene_verts: [n_scene_vert, 3]
def bps_encode_scene(basis_set, scene_verts):
    nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(scene_verts)
    neigh_dist, neigh_ind = nbrs.kneighbors(basis_set)
    # x_bps = scene_verts[neigh_ind].squeeze() - basis_set  # if use delta feature  [n_bps, 3]
    selected_scene_verts = scene_verts[neigh_ind[:,0]]    # scene bps verts  [n_bps, 3]
    x_bps = neigh_dist.transpose((1, 0))  # bps feture, [1, n_bps]
    return x_bps, selected_scene_verts, np.squeeze(neigh_ind)

# basis_set: scene bps verts [n_bps, 3]
# body_verts: [n_body_vert, 3]
def bps_encode_body(basis_set, body_verts):
    nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(body_verts)
    neigh_dist, neigh_ind = nbrs.kneighbors(basis_set)
    x_bps = neigh_dist.transpose((1, 0))      # bps feture, [1, n_bps]
    return x_bps
