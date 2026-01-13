import numpy as np

def extract_and_dump_to_npy(x_file, y_file, z_file, output_prefix="spin"):
    """
    Extract plot data from three files containing x, y, and z spin components.
    Stores coordinates and spin values in NumPy binary files (.npy).

    Args:
        x_file (str): Path to file containing x-components of spin
        y_file (str): Path to file containing y-components of spin
        z_file (str): Path to file containing z-components of spin
        output_prefix (str): Prefix for output files (default: "output")

    Returns:
        dict: Dictionary containing metadata, coordinates, and spin values
    """
    def process_file(file_path):
        """
        Helper function to process a single spin component file.

        Args:
            file_path (str): Path to the input file

        Returns:
            tuple: (coordinates, spin_values, nbnd, nks)
                coordinates: List of 3D coordinates
                spin_values: NumPy array of spin values for this component
                nbnd: Number of bands
                nks: Number of k-points
        """
        # Read and clean the file lines (remove empty lines)
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]

        # Extract metadata from the first line
        metadata_line = lines[0]
        nbnd = int(metadata_line.split('nbnd=')[1].split(',')[0].strip())  # Number of bands
        nks = int(metadata_line.split('nks=')[1].split('/')[0].strip())    # Number of k-points

        # Initialize storage for coordinates and spin values
        coordinates = []
        spin_values = []

        # Start processing from line 1 (after metadata)
        i = 1
        while i < len(lines):
            # Get coordinate (3 values) - only store from first file
            coord = list(map(float, lines[i].split()))

            # Collect all spin values for this coordinate
            component_spins = []
            i += 1  # Move to first spin value line

            # Keep reading spin value lines until we hit another coordinate or end of file
            while i < len(lines) and len(lines[i].split()) != 3:
                # Add all spin values from this line to our list
                component_spins.extend(map(float, lines[i].split()))
                i += 1  # Move to next line

            # Verify we have exactly nbnd spin values
            if len(component_spins) != nbnd:
                raise ValueError(
                    f"Expected {nbnd} spin values at coordinate {coord}, "
                    f"got {len(component_spins)}. Check file format."
                )

            # Add these spin values to our collection
            spin_values.append(component_spins)

            # Only store coordinates from the first file (x-component file)
            if file_path == x_file:
                coordinates.append(coord)

        # Convert spin values to NumPy array and return all data
        return coordinates, np.array(spin_values, dtype=np.float64), nbnd, nks

    # Process each spin component file
    # For x-file, we want all return values (coordinates, spins, nbnd, nks)
    coordinates, x_spins, nbnd, nks = process_file(x_file)

    # For y and z files, we only need the spin values
    # Use _ for values we don't need to capture
    _, y_spins, _, _ = process_file(y_file)
    _, z_spins, _, _ = process_file(z_file)

    # Combine x, y, z components into a 3D array with shape (nks, nbnd, 3)
    # axis=2 means we stack along a new third dimension
    spin_values = np.stack((x_spins, y_spins, z_spins), axis=2)

    # Validate that we have the expected array shape
    if spin_values.shape != (nks, nbnd, 3):
        raise ValueError(
            f"Expected spin_values shape ({nks}, {nbnd}, 3), "
            f"got {spin_values.shape}. Check input files."
        )

    # Save metadata (nbnd and nks) to a .npy file
    np.save(f"{output_prefix}_metadata.npy", np.array([nbnd, nks]))

    # Save coordinates and spin values to a .npy file
    np.save(f"{output_prefix}_data.npy", {
        'coordinates': np.array(coordinates, dtype=np.float64),
        'spin_values': spin_values
    })

    # Return the results in a dictionary
    return {
        'metadata': {'nbnd': nbnd, 'nks': nks},
        'coordinates': np.array(coordinates, dtype=np.float64),
        'spin_values': spin_values
    }

# This block only executes when the script is run directly (not when imported)
if __name__ == "__main__":
    # Example usage - replace with your actual file paths
    result = extract_and_dump_to_npy(
        "outbands.dat.1",  # File with x components of spin
        "outbands.dat.2",  # File with y components of spin
        "outbands.dat.3"   # File with z components of spin
    )

    # Print information about the saved files and data shapes
    print("Metadata saved to: output_metadata.npy")
    print("Data saved to: output_data.npy")
    print(f"\nCoordinates shape: {result['coordinates'].shape}")  # Should be (nks, 3)
    print(f"Spin values shape: {result['spin_values'].shape}")    # Should be (nks, nbnd, 3)

    # Print first coordinate and spin values
    print(f"\nFirst coordinate: {result['coordinates'][0]}")
    print(f"First spin values (first 5 bands): {result['spin_values'][0, :5]}")


