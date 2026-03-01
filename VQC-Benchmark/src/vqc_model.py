import time
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, ZZFeatureMap
from sklearn.metrics import accuracy_score, f1_score
from qiskit_algorithms.utils import algorithm_globals


def create_feature_map(encoding: str, n_qubits: int):
    """Create a feature map circuit for data encoding.

    Feature map complexity is fixed (reps=1) to isolate
    the effect of ansatz depth in experiments.
    """
    if encoding == "angle":
        params = ParameterVector("x", n_qubits)
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.ry(params[i], i)
        return qc
    elif encoding == "zz":
        return ZZFeatureMap(feature_dimension=n_qubits, reps=1)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def create_ansatz(ansatz_name: str, n_qubits: int, reps: int = 1):
    if ansatz_name == "real_amplitudes":
        return RealAmplitudes(num_qubits=n_qubits, reps=reps)
    elif ansatz_name == "efficient_su2":
        return EfficientSU2(num_qubits=n_qubits, reps=reps)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz_name}")


def train_vqc(X_train, y_train, X_test, y_test, n_qubits=4, encoding="angle",
              ansatz_name="real_amplitudes", optimizer_name="cobyla", maxiter=100,
              reps=1, random_seed=None):
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit_machine_learning.algorithms import VQC
    from qiskit.primitives import StatevectorSampler

    # Set Qiskit-internal random seed for reproducibility
    if random_seed is not None:
        algorithm_globals.random_seed = random_seed

    feature_map = create_feature_map(encoding, n_qubits)
    ansatz = create_ansatz(ansatz_name, n_qubits, reps=reps)

    optimizers = {"cobyla": COBYLA(maxiter=maxiter), "spsa": SPSA(maxiter=maxiter)}
    opt = optimizers[optimizer_name]

    sampler = StatevectorSampler()
    vqc = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz, optimizer=opt)

    start = time.time()
    vqc.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred_train = vqc.predict(X_train)
    y_pred_test = vqc.predict(X_test)

    return {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "train_f1": f1_score(y_train, y_pred_train, average="macro"),
        "test_f1": f1_score(y_test, y_pred_test, average="macro"),
        "train_time": train_time,
    }
