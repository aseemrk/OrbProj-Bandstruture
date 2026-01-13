import numpy as np
import matplotlib.pyplot as plt
#import scienceplots 

#plt.style.use(['science','ieee'])
Ha2eV = 27.211407953  # eV/Hartree
e_fermi = 1.8359  # Replace with your Fermi energy value


###################

def extract_second_column_to_2d_array(file_path):
    """
    Extracts second column values from a file and stores them as a 2D array.
    Each new row in the array corresponds to a block of data separated by empty lines.

    Args:
        file_path (str): Path to the input file.

    Returns:
        numpy.ndarray: 2D array of second column values.
    """
    # Initialize a list to store the 2D array
    second_column_2d = []
    current_block = []

    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace

            if not line:  # If the line is empty
                if current_block:  # If current_block is not empty, add it to the 2D array
                    second_column_2d.append(current_block)
                    current_block = []  # Reset current_block for the next block
            else:  # If the line is not empty
                parts = line.split()
                if len(parts) >= 2:  # Ensure there are at least two columns
                    try:
                        current_block.append(float(parts[1]))  # Append the second column value
                    except ValueError:
                        # Skip lines where the second column is not a number
                        continue

    # Add the last block if it's not empty
    if current_block:
        second_column_2d.append(current_block)

    # Convert the list of lists to a 2D numpy array
    second_column_2d_array = np.array(second_column_2d)

    return second_column_2d_array


def extract_x_coordinates_from_file(file_path):
    """
    Extracts x-coordinates of high symmetry points from a file.

    Args:
        file_path (str): Path to the input file containing high symmetry point data.

    Returns:
        list[float]: List of x-coordinates.
    """
    x_coords = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'x coordinate' in line:
                # Split the line into parts and take the last element (x-coordinate)
                parts = line.strip().split()
                x_coord = float(parts[-1])
                x_coords.append(x_coord)
    return x_coords


def extract_first_column_until_break(file_path):
    """
    Extracts the first column from a file until the first empty line.

    Args:
        file_path (str): Path to the input file.

    Returns:
        list: List of values from the first column.
    """
    first_column = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace

            # Stop at the first empty line
            if not line:
                break

            # Split the line into parts
            parts = line.split()

            # If the line has content, extract the first column
            if parts:
                try:
                    # Convert to float (or keep as string if needed)
                    value = float(parts[0])
                    first_column.append(value)
                except ValueError:
                    # Skip lines where the first column is not a number
                    continue

    return first_column



file_path = 'outbands.dat.gnu'  # Replace with your file path
print(f"Loading band structure data from {file_path}")

eigs = extract_second_column_to_2d_array(file_path) - e_fermi
eigs = eigs.T  # Transpose to have shape (nk, nbnd)
print(f"Band structure data shape: {eigs.shape}")

k_path = extract_first_column_until_break(file_path)


overlaps = np.load('proj_data.npy',allow_pickle=True) * 0.5 
#overlaps = np.sum(overlaps,axis=2) * 0.1  # Sum over all atomic states

print(f"Atomic orbitals overlaps loaded")

#colors, alpha = get_color_array(overlaps,eigs) # Get colors based on y-component of spin

fig, ax = plt.subplots(figsize=(3.5,3.5))

for i_band in range(eigs.shape[1]):
    print(f"Plotting band {i_band+1}/{eigs.shape[1]}", end='\r')
    #ax.plot(k_path, eigs[:,i_band],color='gray',linewidth=1.5, zorder=1)
    ax.scatter(k_path, eigs[:,i_band], color='darkblue', s=5, alpha = overlaps[:,i_band,0], zorder=10, edgecolors='none')
    ax.scatter(k_path, eigs[:,i_band], color='gold', s=5, alpha = overlaps[:,i_band,1], zorder=10, edgecolors='none')
    ax.scatter(k_path, eigs[:,i_band], color='cyan', s=5, alpha = overlaps[:,i_band,2], zorder=10, edgecolors='none')


ax.scatter([], [], color='darkblue', s=20, alpha = 1.0, label=r'1D inorganic chains (PbBr$_3$)',edgecolors='none')
ax.scatter([], [], color='gold', s=20, alpha = 1.0, label='Rest of molecule',edgecolors='none')
ax.scatter([], [], color='cyan', s=20, alpha = 1.0, label='Br (molecule)',edgecolors='none')


vbm_indices = np.where(np.abs(eigs) < 0.0001)
n_bnd_vbm = np.max(vbm_indices[1][:])
n_bnd_cbm = n_bnd_vbm + 1

# plot vbm 
k_vbm = np.argmin(np.abs(eigs[:,n_bnd_vbm] - np.max(eigs[:,n_bnd_vbm])))
ax.scatter(k_path[k_vbm], eigs[k_vbm,n_bnd_vbm], color='green', s=50, marker='s', label='VBM',zorder=-50)

# plot cbm 
k_cbm = np.argmin(np.abs(eigs[:,n_bnd_cbm] - np.min(eigs[:,n_bnd_cbm])))
ax.scatter(k_path[k_cbm], eigs[k_cbm,n_bnd_cbm], color='deepskyblue', marker='s', label = 'CBM', s=50,zorder=-50)

print('Band gap = ',eigs[k_cbm,n_bnd_cbm] - eigs[k_vbm,n_bnd_vbm])

file_high_sym = 'bands_pp.out'  # Replace with your file path
x_coords = extract_x_coordinates_from_file(file_high_sym)
x_coords_labels = [r'$\Gamma$', 'Z', 'D', 'B', r'$\Gamma$', 'A', 'E', 'Z','M', 'H', r'$\Gamma$','Y','C', 'Z']  # Replace with your actual labels

for x in x_coords:
    ax.axvline(x, color='lightgray', linestyle='dashed', linewidth=0.5, zorder=-10)
    

plt.xticks(x_coords, x_coords_labels,fontsize=7)
plt.xlim(x_coords[0], x_coords[-1])
plt.ylim(-1.5, 4.5)
plt.ylabel('Energy (eV)')
plt.xlabel('Wave vector (k)')
plt.legend()
plt.savefig('atom_proj.png', dpi = 400, bbox_inches='tight')
