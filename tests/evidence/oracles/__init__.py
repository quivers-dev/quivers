"""Oracles that import nothing from the library.

Each oracle takes plain data (tuples, dictionaries, strings, rationals)
and computes its answer by direct enumeration or from a closed form. An
oracle never imports the code it is used to check;
``tests/evidence/test_oracles.py`` reads the import statements and fails
when one does.
"""
