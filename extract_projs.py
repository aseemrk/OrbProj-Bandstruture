import numpy as np
import sys
import xml.etree.ElementTree as ET

# Input file names
dosout = 'pdos.out'
atomic_states = [['19Pb','20Pb','27Br','28Br','29Br','30Br','31Br','32Br'],
                 ['H','C','N'],['33Br','34Br']]  # List of atomic states to project onto
atom_proj_file = 'atomic_proj.xml'
SOC = True  # Whether spin-orbit coupling is considered

def parse_dos_file(dosout, SOC):
    """
    Parse a DOS file to extract state information.

    Args:
        dosout (str): Path to the DOS file
        SOC (bool): Whether spin-orbit coupling is considered (default: False)

    Returns:
        tuple: (natomwfc, nbnd, nkpt, list_states)
            - natomwfc: Number of atomic wavefunctions
            - nbnd: Number of bands
            - nkpt: Number of k-points
            - list_states: List of state information [state_index, state_string]
    """
    with open(dosout, 'r') as dos_file:
        lines = dos_file.readlines()

    list_states = []
    natomwfc = None
    nbnd = None
    nkpt = None

    # Parse header and state lines
    for line in lines:
        if 'natomwfc' in line:
            natomwfc = int(line.split()[-1])
        elif 'nbnd' in line:
            nbnd = int(line.split()[-1])
        elif 'nkstot' in line:
            nkpt = int(line.split()[-1])
        elif 'state # ' in line:
            # Extract state index and state string
            state_index = int(line.split(':')[0][12:16]) - 1
            if SOC:
                state_string = line.split('(')[0][-4:-1] + line.split('(')[1][0:2] + ' l_' + line.split('(')[2][2] + '_j_' + line.split('(')[2][6:9]
                #print(state_string)
            else:
                state_string = line.split('(')[1][0:2] + ' l_' + line.split('(')[2][2] + '_m_' + line.split('(')[2][7:8]
            state_string = state_string.replace(" ", "_")
            list_states.append([state_index, state_string])

    return natomwfc, nbnd, nkpt, list_states



def parse_atom_proj_file(atom_proj_file):
    """
    Parse an atom projection XML file and extract eigenvalues, and array DOS.

    Args:
        atom_proj_file (str): Path to the atom projection XML file

    Returns:
        tuple: (eigs, array_dos)
            - eigs: Array of eigenvalues with shape (nkpt, nbnd)
            - array_dos: Array of DOS with shape (nkpt, nbnd, natomwfc)
    """
    # Parse the XML file
    tree = ET.parse(atom_proj_file)
    root = tree.getroot()
    # Extract necessary attributes from header
    header = root.find('HEADER')
    nbnd = int(header.get('NUMBER_OF_BANDS', 0))
    nkpt = int(header.get('NUMBER_OF_K-POINTS', 0))
    natomwfc = int(header.get('NUMBER_OF_ATOMIC_WFC', 0))    
    # Initialize arrays for eigenvalues, DOS, 
    eigs = np.zeros((nkpt, nbnd))
    array_dos = np.zeros((nkpt, nbnd, natomwfc))

    print(f"Reading {atom_proj_file}")

    # Process each k-point
    for k in range(nkpt):
        print(f"Loading data for k-point {k+1}/{nkpt}", end='\r')
        # Extract eigenvalues (convert from Hartree to eV)
        eigs[k,:] = np.fromstring(root[1][3*k+1].text, dtype=float, sep=' ') * 13.605698066

        # Process each atomic wavefunction
        for j in range(natomwfc):
            # Extract real and imaginary parts of overlaps
            overlaps = np.fromstring(root[1][k*3+2][j].text, dtype=float, sep=' ')
            real_overlap = overlaps[0::2]
            imag_overlap = overlaps[1::2]

            # Calculate and store DOS for this atomic wavefunction
            array_dos[k,:,j] = (real_overlap**2 + imag_overlap**2) #* float(root[1][k*3].attrib['Weight'])

    return eigs, array_dos

# Parse the DOS and atomic projection files
natomwfc, nbnd, nkpt, list_states = parse_dos_file(dosout, SOC)
eigs, array_dos = parse_atom_proj_file(atom_proj_file)

print(f"Parsed DOS file: natomwfc={natomwfc}, nbnd={nbnd}, nkpt={nkpt}")
#print(list_states)

# Initialize overlaps array: shape (nkpt, nbnd, number of atomic states)
overlaps = np.zeros((nkpt, nbnd, len(atomic_states)))

# For each atomic state, sum the DOS contributions from relevant states
for k in range(nkpt):
    print(f"Calculating overlaps for k-point {k+1}/{nkpt}", end='\r')
    for a,atomic_state in enumerate(atomic_states):
        states_in_weight = np.zeros((nkpt,natomwfc),dtype=bool)
        for substate in atomic_state:
            for i in range(natomwfc):
                # Mark states that match the current atomic state
                if substate in list_states[i][1]:
                    states_in_weight[k,i] = True
        # Sum the DOS for the selected states at this k-point
        overlaps[k,:,a] = np.sum(array_dos[k,...],axis=1,where=states_in_weight[k,:])

# Save the overlaps array to a .npy file
np.save('proj_data.npy', overlaps)

# --- Save meta data as .npy ---
meta = {
    "nbnd": nbnd,  # Number of bands
    "nkpt": nkpt,  # Number of k-points
    "natomwfc": natomwfc,  # Number of atomic wavefunctions
    "atomic_states": atomic_states,  # List of atomic states
    "list_states": np.array(list_states, dtype=object),  # List of state indices and strings
    "eigs_shape": eigs.shape,  # Shape of eigenvalues array
    "array_dos_shape": array_dos.shape,  # Shape of array_dos
}

# Save the meta data dictionary as a .npy file
np.save('proj_data_meta.npy', meta)
