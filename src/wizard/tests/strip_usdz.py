# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Script for stripping away all content of a USDZ file besides the metadata. This is useful for preparing lightweight
artifacts for testing.
"""

import argparse
import glob
import io
import zipfile

import yaml


def strip_usdz(usdz_bytes: bytes) -> bytes:
    # Read the original ZIP file from bytes
    with zipfile.ZipFile(io.BytesIO(usdz_bytes), "r") as zip_ref:
        metadata = zip_ref.read("metadata.yaml")

    # Parse the metadata
    metadata_dict = yaml.safe_load(metadata)
    metadata_dict["version_string"] = f"stripped-{metadata_dict['version_string']}"
    metadata_dict["uuid"] = f"stripped-{metadata_dict['uuid']}"

    # Write the modified metadata back to bytes
    metadata_str = yaml.dump(metadata_dict)
    metadata_bytes = metadata_str.encode("utf-8")

    # Create a new ZIP file in memory
    output_zip_bytes = io.BytesIO()
    with zipfile.ZipFile(output_zip_bytes, "w", zipfile.ZIP_STORED) as zip_out:
        zip_out.writestr("metadata.yaml", metadata_bytes)

    return output_zip_bytes.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usdz_glob", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    usdz_files = glob.glob(args.usdz_glob)

    for file in usdz_files:
        with open(file, "rb") as usdz_file:
            stripped_bytes = strip_usdz(usdz_file.read())

        output_path = f"{args.output_dir}/{file.split('/')[-1]}"
        print(f"Writing stripped USDZ to {output_path}")
        with open(output_path, "wb") as output_file:
            output_file.write(stripped_bytes)
