class ResidueMetadata:
    """Metadata for residues and atoms."""

    ATOM_TYPES: list[str] = ["C", "O", "N", "F", "S", "H"]
    ATOM_CODES: list[str] = ["C", "O", "N", "S", "CA", "CB", "H"]
    RESIDUE_CODES: list[str] = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLU",
        "GLN",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "ACE",
        "NME",
        "UNK",
    ]

    # One to three letter code mapping
    AA_3CODES: dict[str, str] = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "E": "GLU",
        "Q": "GLN",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }

    # Three to one letter code mapping
    AA_1CODES: dict[str, str] = {v: k for k, v in AA_3CODES.items()}


def encode_atom_type(atom_type: str) -> int:
    """Encode atom symbol (eg. C) as an integer."""
    if atom_type in ResidueMetadata.ATOM_TYPES:
        return ResidueMetadata.ATOM_TYPES.index(atom_type)
    else:
        return len(ResidueMetadata.ATOM_TYPES)


def encode_atom_code(atom_code: str) -> int:
    """Encode atom code (eg. CA) as an integer."""
    if atom_code in ResidueMetadata.ATOM_CODES:
        return ResidueMetadata.ATOM_CODES.index(atom_code)
    else:
        return len(ResidueMetadata.ATOM_CODES)


def encode_residue(residue_name: str) -> int:
    """Encode residue name as an integer."""
    if residue_name.startswith("Me+"):
        return len(ResidueMetadata.RESIDUE_CODES) + encode_residue(residue_name[len("Me+") :])
    if residue_name in ResidueMetadata.RESIDUE_CODES:
        return ResidueMetadata.RESIDUE_CODES.index(residue_name)
    else:
        raise ValueError(f"Invalid residue name: {residue_name}. Valid names are: {ResidueMetadata.RESIDUE_CODES}")
        return len(ResidueMetadata.RESIDUE_CODES)


def convert_to_three_letter_code(aa: str) -> str:
    """Convert one-letter amino acid code to three-letter code."""
    if aa.startswith("Me+"):
        return "Me+" + convert_to_three_letter_code(aa[len("Me+") :])  # Return the rest of the string after "Me"

    aa = aa.upper()
    if len(aa) == 1:
        if aa not in ResidueMetadata.AA_3CODES:
            raise ValueError(f"Invalid one-letter amino acid code: {aa}")
        return ResidueMetadata.AA_3CODES[aa]
    elif len(aa) == 3:
        if aa not in ResidueMetadata.AA_1CODES.values():
            raise ValueError(f"Invalid three-letter amino acid code: {aa}")
        return aa
    else:
        raise ValueError(f"Invalid amino acid code length: {aa}")


def convert_to_three_letter_codes(peptide: str) -> str:
    """Convert peptides with one-letter amino acid codes to peptides with three-letter codes."""
    if "_" in peptide:
        return peptide
    return "_".join([convert_to_three_letter_code(aa) for aa in peptide])


def convert_to_one_letter_code(aa: str) -> str:
    """Convert three-letter amino acid code to one-letter code."""
    aa = aa.upper()
    if len(aa) == 1:
        if aa not in ResidueMetadata.AA_3CODES:
            raise ValueError(f"Invalid one-letter amino acid code: {aa}")
        return aa
    elif len(aa) == 3:
        if aa not in ResidueMetadata.AA_1CODES:
            raise ValueError(f"Invalid three-letter amino acid code: {aa}")
        return ResidueMetadata.AA_1CODES[aa]
    else:
        raise ValueError(f"Invalid amino acid code length: {aa}")


def convert_to_one_letter_codes(peptide: str) -> str:
    """Convert peptides with three-letter amino acid codes to peptides with one-letter codes."""
    if "_" not in peptide:
        return peptide
    return "".join([convert_to_one_letter_code(aa) for aa in peptide.split("_")])
