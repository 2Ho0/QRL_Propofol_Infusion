import sys
import os
import pennylane as qml
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

# Setup path to import src
sys.path.append('/home/mwilliam/QRL_Propofol_Infusion')

from src.models.vqc import VariationalQuantumCircuit

def main():
    n_qubits = 2
    n_layers = 4
    
    # Instantiate the actual VQC from the project
    vqc = VariationalQuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
    
    # Data Re-Uploading: layer_inputs shape = [n_layers, n_qubits]
    layer_inputs = np.zeros((n_layers, n_qubits))
    weights = vqc.get_initial_weights().detach().numpy()
    
    # Convert to JAX arrays
    layer_inputs_jax = jnp.array(layer_inputs, dtype=jnp.float32)
    weights_jax = jnp.array(weights, dtype=jnp.float32)
    
    try:
        # Extract the unjitted QNode circuit from JAX jit for matplotlib drawing
        try:
            unjitted_circuit = vqc.circuit.__wrapped__
            fig, ax = qml.draw_mpl(unjitted_circuit)(layer_inputs_jax, weights_jax)
        except AttributeError:
            fig, ax = qml.draw_mpl(vqc.circuit)(layer_inputs_jax, weights_jax)
        
        # Save figure
        output_path = '/home/mwilliam/QRL_Propofol_Infusion/quantum_circuit_vqc.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved circuit diagram to {output_path}")
    except Exception as e:
        print(f"Error while drawing: {e}")

if __name__ == "__main__":
    main()
