"""
Unit tests for Cryptographic Encryption Vault and PII Scrubber
"""

import os
import unittest
from modules.encryption import DocumentVault, PIIScrubber


class TestDocumentVault(unittest.TestCase):
    def setUp(self):
        self.test_vault_dir = ".test_vault"
        self.vault = DocumentVault(vault_dir=self.test_vault_dir)

    def tearDown(self):
        self.vault.purge_vault()
        if os.path.exists(self.test_vault_dir):
            try:
                os.rmdir(self.test_vault_dir)
            except Exception:
                pass

    def test_encryption_and_decryption(self):
        raw_data = b"Confidential Corporate Strategy Document - 2026"
        filename = "strategy.txt"

        meta = self.vault.encrypt_and_store(filename, raw_data)

        # Ciphertext file should exist on disk
        self.assertTrue(os.path.exists(meta["vault_path"]))

        # Ciphertext should NOT match plaintext
        with open(meta["vault_path"], "rb") as f:
            stored_cipher = f.read()
        self.assertNotEqual(stored_cipher, raw_data)

        # Decrypt in memory
        decrypted = self.vault.decrypt_in_memory(filename)
        self.assertEqual(decrypted, raw_data)

    def test_integrity_verification(self):
        raw_data = b"Patient Medical History #99281"
        filename = "patient.txt"
        meta = self.vault.encrypt_and_store(filename, raw_data)

        # Corrupt the ciphertext file on disk
        with open(meta["vault_path"], "wb") as f:
            f.write(b"corrupted_bytes_tampered_payload")

        # In-memory decryption must raise an exception upon tamper detection
        with self.assertRaises(Exception):
            self.vault.decrypt_in_memory(filename)

    def test_pii_scrubber(self):
        sample = "Contact Dr. Priya at priya.shukla@example.com or phone +1-555-123-4567. SSN: 123-45-6789."
        scrubbed, stats = PIIScrubber.redact(sample)

        self.assertNotIn("priya.shukla@example.com", scrubbed)
        self.assertIn("[REDACTED_EMAIL]", scrubbed)
        self.assertNotIn("123-45-6789", scrubbed)
        self.assertIn("[REDACTED_SSN]", scrubbed)
        self.assertEqual(stats["emails"], 1)
        self.assertEqual(stats["ssn"], 1)


if __name__ == "__main__":
    unittest.main()
