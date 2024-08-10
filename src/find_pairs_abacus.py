import numpy as np
from scipy.spatial import cKDTree
import h5py
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import os

def compute_pairs_FOF_abacus_box(phaseID=0, vcirc_threshold=200.0, exclusion_factor=3.0):
    # Define the path to the halo catalog
    basePath = f"/global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_base_c000_ph{phaseID:03d}/halos/z0.100/"
    fields = ["id", "N", "x_L2com", "v_L2com", "vcirc_max_L2com"]
    
    print("Started reading the data")
    cat = CompaSOHaloCatalog(os.path.join(basePath), fields=fields)
    print("Finished reading the data")
    
    BoxSize = cat.header['BoxSizeHMpc']
    print('BoxSize', BoxSize)
    
    print("Started Vcirc selection", len(cat.halos))
    ii = cat.halos['vcirc_max_L2com'] > vcirc_threshold  # in units of km/s
    cat.halos = cat.halos[ii]
    print("Finished Vcirc selection", len(cat.halos))
    
    S_pos = cat.halos['x_L2com']
    S_pos = (S_pos + 0.5*BoxSize)%BoxSize
    S_vel = cat.halos['v_L2com']
    S_vmax = cat.halos['vcirc_max_L2com']
    S_mass = cat.halos['N']
    S_id = cat.halos['id']
    n_S = len(S_pos)
    print(f"Number of halos selected: {n_S}")
    
    def find_pairs(positions, box_size):
        tree = cKDTree(positions, boxsize=box_size)
        pairs = set()
    
        for i in range(len(positions)):
            dist, idx = tree.query(positions[i], k=2, distance_upper_bound=box_size/2)
            if idx[1] != i and tree.query(positions[idx[1]], k=2)[1][1] == i:
                # Ensure i < j to avoid duplicates
                pair = (min(i, idx[1]), max(i, idx[1]))
                pairs.add(pair)
    
        return list(pairs)

    def check_isolation(pair, positions, vcirc_max, tree, exclusion_factor):
        i, j = pair
        pair_distance = np.linalg.norm(positions[i] - positions[j])
        search_radius = pair_distance * exclusion_factor
        
        for idx in [i, j]:
            neighbors = tree.query_ball_point(positions[idx], r=search_radius)
            if any(vcirc_max[n] > vcirc_max[idx] for n in neighbors if n != i and n != j):
                return False
        
        return True

    print("Started pair finding")
    tree = cKDTree(S_pos, boxsize=BoxSize)
    pairs = find_pairs(S_pos, BoxSize)
    print(f"Found {len(pairs)} potential pairs")

    print("Started isolation check")
    isolated_pairs = [pair for pair in pairs if check_isolation(pair, S_pos, S_vmax, tree, exclusion_factor)]
    print(f"Found {len(isolated_pairs)} isolated pairs")

    halo_A_id = np.array([pair[0] for pair in isolated_pairs], dtype=int)
    halo_B_id = np.array([pair[1] for pair in isolated_pairs], dtype=int)

    filename = f"../data/pairs_AbacusSummit_base_c000_ph{phaseID:03d}_z0.100.hdf5"
    print(f"Started writing data to {filename}")
    
    with h5py.File(filename, 'w') as h5f:
        h5f.create_dataset('pos_A', data=S_pos[halo_A_id])
        h5f.create_dataset('pos_B', data=S_pos[halo_B_id])
        h5f.create_dataset('mass_A', data=S_mass[halo_A_id])
        h5f.create_dataset('mass_B', data=S_mass[halo_B_id])
        h5f.create_dataset('pos_G', data=S_pos)
        h5f.create_dataset('vel_A', data=S_vel[halo_A_id])
        h5f.create_dataset('vel_B', data=S_vel[halo_B_id])
        h5f.create_dataset('vel_G', data=S_vel)
        h5f.create_dataset('vmax_A', data=S_vmax[halo_A_id])
        h5f.create_dataset('vmax_B', data=S_vmax[halo_B_id])
        h5f.create_dataset('vmax_G', data=S_vmax)
        h5f.create_dataset('halo_A_id', data=S_id[halo_A_id])
        h5f.create_dataset('halo_B_id', data=S_id[halo_B_id])
    
    print("Finished writing data")

if __name__ == "__main__":
    compute_pairs_FOF_abacus_box()