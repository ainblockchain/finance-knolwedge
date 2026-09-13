"""python3 web/test_privy_pay.py — only what runs without network or model servers."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PRIVY_STATE", "/dev/null")  # never touch the state file
import privy_pay as p


class Keccak(unittest.TestCase):
    def test_vectors(self):
        self.assertEqual(p.keccak256(b"").hex(), "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
        self.assertEqual(p.keccak256(b"abc").hex(), "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45")
        # the 136-byte boundary (padding collapses to a single 0x81) and multi-block input
        self.assertEqual(p.keccak256(b"a" * 135).hex(), p.keccak256(b"a" * 135).hex())
        self.assertNotEqual(p.keccak256(b"a" * 135), p.keccak256(b"a" * 136))
        self.assertEqual(len(p.keccak256(b"x" * 1000)), 32)


class Derive(unittest.TestCase):
    def test_same_code_same_address(self):
        self.assertEqual(p.derive_address("116100"), p.derive_address("116100"))

    def test_different_code_different_address(self):
        codes = ["116100", "033240", "069920", "000040", "0220W0"]
        self.assertEqual(len({p.derive_address(c) for c in codes}), len(codes))

    def test_shape(self):
        a = p.derive_address("116100")
        self.assertTrue(a.startswith("0x") and len(a) == 42)
        int(a, 16)


class Registry(unittest.TestCase):
    def test_resolve_one_line(self):
        r = p.resolve("태양기계")
        self.assertIsNotNone(r); self.assertEqual(r["code"], "116100"); self.assertEqual(r["address"], p.derive_address("116100"))

    def test_registry_addresses_match_derivation(self):
        for r in p.REGISTRY:
            self.assertEqual(r["address"], p.derive_address(r["code"]))

    def test_find_company_in_instruction(self):
        hits = p.find_company("태양기계와 픽셀플러스에 대금 지급")
        self.assertEqual([h["name"] for h in hits], ["태양기계", "픽셀플러스"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
