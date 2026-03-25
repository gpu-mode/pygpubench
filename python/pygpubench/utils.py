import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class DeterministicContext:
    def __init__(self):
        self.allow_tf32 = None
        self.deterministic = None
        self.cublas = None

    def __enter__(self):
        import torch
        self.cublas = os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')
        self.allow_tf32 = torch.backends.cudnn.allow_tf32
        self.deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import torch
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.deterministic = self.deterministic
        torch.use_deterministic_algorithms(False)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = self.cublas


def decrypt_benchmark_result(data: bytes, key: bytes) -> str:
    NONCE_LEN = 12
    TAG_LEN   = 16

    if len(data) < NONCE_LEN + TAG_LEN:
        raise ValueError("Invalid benchmark result: too short")

    nonce      = data[:NONCE_LEN]
    tag        = data[NONCE_LEN:NONCE_LEN + TAG_LEN]
    ciphertext = data[NONCE_LEN + TAG_LEN:]

    aesgcm    = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext + tag, None)
    except Exception as E:
        raise RuntimeError("Could not decrypt benchmark result: Have they been tampered with?") from E
    return plaintext.decode("utf-8")
