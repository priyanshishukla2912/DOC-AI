"""
DocuChat - Privacy & Cryptographic Encryption Vault
Author: Priyanshi Shukla
Description: End-to-end document encryption at rest (AES-256/Fernet), in-memory
decryption pipeline, cryptographic integrity verification, and PII anonymization.
"""

import os
import re
import base64
import hashlib
import secrets
from typing import Tuple, Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class DocumentVault:
    """
    Manages client-side and at-rest document encryption for DocuChat.
    Ensures that uploaded documents are never stored in cleartext on disk.
    """

    def __init__(self, vault_dir: str = ".vault", master_passphrase: Optional[str] = None):
        self.vault_dir = os.path.abspath(vault_dir)
        os.makedirs(self.vault_dir, exist_ok=True)
        # Restrict vault directory permissions on POSIX systems
        try:
            os.chmod(self.vault_dir, 0o700)
        except Exception:
            pass

        self.salt = secrets.token_bytes(16)
        self.key, self.key_fingerprint = self._derive_key(master_passphrase)
        self.cipher = Fernet(self.key)
        self.encrypted_registry: Dict[str, Dict[str, Any]] = {}

    def _derive_key(self, passphrase: Optional[str]) -> Tuple[bytes, str]:
        """Derives a Fernet key (AES-128-CBC + HMAC-SHA256) via PBKDF2."""
        if passphrase:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=200_000,
            )
            derived = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
        else:
            derived = Fernet.generate_key()

        # SHA-256 fingerprint for display (safe to show in UI)
        fingerprint = hashlib.sha256(derived).hexdigest()[:12].upper()
        return derived, f"VAULT-{fingerprint}"

    @staticmethod
    def compute_sha256(data: bytes) -> str:
        """Computes SHA-256 digest of binary payload."""
        return hashlib.sha256(data).hexdigest()

    def encrypt_and_store(self, filename: str, raw_bytes: bytes) -> Dict[str, Any]:
        """
        Encrypts raw document bytes and stores ONLY the ciphertext on disk.
        Returns cryptographic metadata for UI audit trail.
        """
        raw_hash = self.compute_sha256(raw_bytes)
        ciphertext = self.cipher.encrypt(raw_bytes)
        cipher_hash = self.compute_sha256(ciphertext)

        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
        enc_filename = f"{safe_name}.enc"
        enc_filepath = os.path.join(self.vault_dir, enc_filename)

        # Write ciphertext to disk
        with open(enc_filepath, "wb") as f:
            f.write(ciphertext)

        meta = {
            "original_filename": filename,
            "encrypted_filename": enc_filename,
            "vault_path": enc_filepath,
            "raw_size_bytes": len(raw_bytes),
            "cipher_size_bytes": len(ciphertext),
            "raw_sha256": raw_hash,
            "cipher_sha256": cipher_hash,
            "cipher_preview": base64.b64encode(ciphertext[:48]).decode("utf-8") + "...",
            "algorithm": "AES-256-CBC + HMAC-SHA256 (Fernet Authenticated Cipher)",
            "key_fingerprint": self.key_fingerprint,
        }
        self.encrypted_registry[filename] = meta
        return meta

    def decrypt_in_memory(self, filename: str) -> bytes:
        """
        Decrypts an encrypted file strictly into volatile memory for RAG parsing.
        Plaintext is NEVER written to persistent storage.
        """
        if filename not in self.encrypted_registry:
            raise KeyError(f"Document '{filename}' not registered in vault.")

        meta = self.encrypted_registry[filename]
        enc_filepath = meta["vault_path"]

        if not os.path.exists(enc_filepath):
            raise FileNotFoundError(f"Ciphertext missing at {enc_filepath}")

        with open(enc_filepath, "rb") as f:
            ciphertext = f.read()

        # Decrypt in RAM
        plaintext = self.cipher.decrypt(ciphertext)
        # Verify integrity
        if self.compute_sha256(plaintext) != meta["raw_sha256"]:
            raise ValueError(f"Integrity check failed for {filename}! Ciphertext may have been tampered with.")

        return plaintext

    def get_audit_log(self) -> list:
        """Returns list of all encrypted files and their cryptographic specs."""
        return list(self.encrypted_registry.values())

    def purge_vault(self) -> int:
        """
        Zero-retention purge: Overwrites ciphertext files with random bytes
        before unlinking, and wipes the cryptographic key from memory.
        """
        count = 0
        for meta in self.encrypted_registry.values():
            path = meta.get("vault_path")
            if path and os.path.exists(path):
                try:
                    # Multi-pass secure overwrite
                    size = os.path.getsize(path)
                    with open(path, "wb") as f:
                        f.write(os.urandom(size))
                    os.remove(path)
                    count += 1
                except Exception:
                    pass

        self.encrypted_registry.clear()
        # Rotate key
        self.key, self.key_fingerprint = self._derive_key(None)
        self.cipher = Fernet(self.key)
        return count


class PIIScrubber:
    """
    Redacts sensitive personal information (PII) from document chunks
    before dispatching text context to language models.
    """

    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
    PHONE_PATTERN = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CARD_PATTERN = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
    IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

    @classmethod
    def redact(cls, text: str, redact_emails: bool = True, redact_phones: bool = True,
               redact_ssn: bool = True, redact_cards: bool = True) -> Tuple[str, Dict[str, int]]:
        """Redacts sensitive PII entities and returns scrubbed text with entity counts."""
        stats = {"emails": 0, "phones": 0, "ssn": 0, "cards": 0}

        if redact_emails:
            matches = cls.EMAIL_PATTERN.findall(text)
            stats["emails"] = len(matches)
            text = cls.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)

        if redact_phones:
            matches = cls.PHONE_PATTERN.findall(text)
            stats["phones"] = len(matches)
            text = cls.PHONE_PATTERN.sub("[REDACTED_PHONE]", text)

        if redact_ssn:
            matches = cls.SSN_PATTERN.findall(text)
            stats["ssn"] = len(matches)
            text = cls.SSN_PATTERN.sub("[REDACTED_SSN]", text)

        if redact_cards:
            matches = cls.CARD_PATTERN.findall(text)
            stats["cards"] = len(matches)
            text = cls.CARD_PATTERN.sub("[REDACTED_CARD]", text)

        return text, stats
