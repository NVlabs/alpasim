# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

from __future__ import annotations

import asyncio
import datetime
import io
import logging
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Any

import boto3
import yaml
from botocore.client import BaseClient
from filelock import FileLock
from typing_extensions import Self

logger = logging.getLogger("alpasim_wizard")


@dataclass
class S3Path:
    """
    Compared to Swiftstack, S3 API assumes the first part of the path is the bucket name.
    Swiftstack: uri=bucket/some/key
    S3: bucket=bucket, key=some/key
    """

    bucket: str
    key: str

    def to_swiftstack(self) -> str:
        return f"{self.bucket}/{self.key}"

    @classmethod
    def from_swiftstack(cls, swiftstack_path: str) -> Self:
        parts = swiftstack_path.split("/", 1)
        return cls(bucket=parts[0], key=parts[1])


class S3SeekableReader(io.RawIOBase):
    """A file-like object that reads from an S3 object using range requests to avoid downloading the entire object."""

    def __init__(self, s3_path: S3Path, client: BaseClient) -> None:
        self.s3_path = s3_path
        self.client = client
        self.size = self._get_size()
        self.pos = 0

    def _get_size(self) -> int:
        head = self.client.head_object(Bucket=self.s3_path.bucket, Key=self.s3_path.key)
        return head["ContentLength"]

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            newpos = offset
        elif whence == io.SEEK_CUR:
            newpos = self.pos + offset
        elif whence == io.SEEK_END:
            newpos = self.size + offset
        else:
            raise ValueError("Invalid whence")
        if newpos < 0:
            raise ValueError("Negative seek position")
        self.pos = newpos
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if n < 0:
            n = self.size - self.pos
        end = self.pos + n - 1
        # Perform a ranged GET request.
        response = self.client.get_object(
            Bucket=self.s3_path.bucket,
            Key=self.s3_path.key,
            Range=f"bytes={self.pos}-{end}",
        )
        data = response["Body"].read()
        self.pos += len(data)
        return data

    def readable(self) -> bool:
        return True


@dataclass
class S3ObjectMetadata:
    """Metadata for an object in S3 in a typed format."""

    path: S3Path
    last_modified: datetime.datetime
    etag: str
    size: int
    storage_class: str

    @classmethod
    def from_dict(cls, bucket: str, obj_dict: dict[str, Any]) -> S3ObjectMetadata:
        return cls(
            path=S3Path(bucket=bucket, key=obj_dict["Key"]),
            last_modified=obj_dict["LastModified"],
            etag=obj_dict["ETag"],
            size=obj_dict["Size"],
            storage_class=obj_dict["StorageClass"],
        )


class S3Connection:
    """A wrapper for the boto3 S3 client that provides some convenience methods."""

    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str,
        endpoint_url: str,
    ) -> None:
        self.session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        self.client = self.session.client(
            "s3",
            endpoint_url=endpoint_url,
        )

    @classmethod
    def from_env_vars(cls) -> S3Connection:
        """Creates an S3Connection from environment variables (uses ALPAMAYO_S3_SECRET)."""
        return cls(
            aws_access_key_id="team-alpamayo",
            aws_secret_access_key=os.environ["ALPAMAYO_S3_SECRET"],
            region_name="us-east-1",
            endpoint_url="https://pdx.s8k.io",
        )

    def list_objects(self, path: S3Path) -> list[S3ObjectMetadata]:
        """Recursively lists all objects under a path in S3 using a paginator."""
        paginator = self.client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=path.bucket, Prefix=path.key)

        objects: list[S3ObjectMetadata] = []
        for page in page_iterator:
            if "Contents" in page:
                objects.extend(
                    S3ObjectMetadata.from_dict(path.bucket, obj_dict)
                    for obj_dict in page["Contents"]
                )
        return objects

    def read_usdz_metadata(self, obj: S3ObjectMetadata) -> USDZMetadata:
        """Reads the metadata.yaml file from a USDZ object in S3."""
        reader = S3SeekableReader(obj.path, self.client)
        with zipfile.ZipFile(reader, "r") as zf:
            with zf.open("metadata.yaml") as f:
                yaml_dict = yaml.safe_load(f)

        return USDZMetadata.from_raw(obj, yaml_dict)

    def check_usdz_has_mesh_ground(self, obj: S3ObjectMetadata) -> bool:
        """Checks if a USDZ file contains mesh_ground.ply."""
        try:
            reader = S3SeekableReader(obj.path, self.client)
            with zipfile.ZipFile(reader, "r") as zf:
                return "mesh_ground.ply" in zf.namelist()
        except Exception:
            return False

    def _maybe_download_object(self, s3_path: S3Path, local_path: str) -> None:
        with FileLock(f"{local_path}.lock", mode=0o666):
            if os.path.exists(local_path):
                return  # TODO: check a checksum or something

            logger.debug(
                f"Starting downloading {s3_path.to_swiftstack()} to {local_path}"
            )
            # download_file uses a temporary file name so we don't need to worry about partial downloads
            try:
                self.client.download_file(s3_path.bucket, s3_path.key, local_path)
                logger.debug(
                    f"Finished downloading {s3_path.to_swiftstack()} to {local_path}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to download {s3_path.to_swiftstack()} to {local_path}"
                )
                os.remove(local_path)
                raise e

    async def maybe_download_object(self, s3_path: S3Path, local_path: str) -> None:
        """Downloads an object from S3 to a local path. Skips the operation if it already exists."""
        return await asyncio.to_thread(self._maybe_download_object, s3_path, local_path)

    def _upload_object(self, local_path: str, s3_path: S3Path) -> None:
        with FileLock(f"{local_path}.lock", mode=0o666):
            logger.debug(
                f"Starting uploading {local_path} to {s3_path.to_swiftstack()}"
            )
            self.client.upload_file(local_path, s3_path.bucket, s3_path.key)
            logger.debug(
                f"Finished uploading {local_path} to {s3_path.to_swiftstack()}"
            )

    async def upload_object(self, local_path: str, s3_path: S3Path) -> None:
        """Uploads an object from a local path to S3."""
        return await asyncio.to_thread(self._upload_object, local_path, s3_path)


NRE_GIT_SHA_RE = re.compile(r"-(\w+)$")


@dataclass
class USDZMetadata:
    uuid: str
    scene_id: str
    nre_version_string: str
    nre_git_sha: str
    dataset_hash: str
    camera_ids: str
    lidar_ids: str

    # these two are strings because kratos doesnt support 64-bit integers...
    scene_start_time: str
    scene_end_time: str

    psnr: float
    training_date: datetime.datetime

    s3_etag: str
    s3_size: int
    s3_last_modified: datetime.datetime
    ss_path: str

    @classmethod
    def from_raw(
        cls, s3_metadata: S3ObjectMetadata, usdz_metadata: dict[str, Any]
    ) -> Self:
        sensors: dict[str, list[str]] = usdz_metadata["sensors"]
        camera_ids = ",".join(sensors.get("camera_ids", []))
        lidar_ids = ",".join(sensors.get("lidar_ids", []))

        time_range: dict[str, int] = usdz_metadata["time_range"]
        scene_start_time = str(time_range["start"])
        scene_end_time = str(time_range["end"])

        training_step_outputs: dict[str, float] = {
            key: float(value)
            for key, value in usdz_metadata["training_step_outputs"].items()
        }
        psnr = training_step_outputs.get("psnr", float("nan"))

        training_date = datetime.datetime.strptime(
            usdz_metadata["training_date"], "%Y-%m-%d"
        )

        version_string = usdz_metadata["version_string"]
        nre_git_sha_match = NRE_GIT_SHA_RE.search(version_string)
        nre_git_sha = nre_git_sha_match.group(1) if nre_git_sha_match else ""

        return cls(
            dataset_hash=usdz_metadata["dataset_hash"],
            scene_id=usdz_metadata["scene_id"],
            uuid=usdz_metadata["uuid"],
            nre_version_string=usdz_metadata["version_string"],
            nre_git_sha=nre_git_sha,
            camera_ids=camera_ids,
            lidar_ids=lidar_ids,
            scene_start_time=scene_start_time,
            scene_end_time=scene_end_time,
            psnr=psnr,
            training_date=training_date,
            ss_path=s3_metadata.path.to_swiftstack(),
            s3_etag=s3_metadata.etag,
            s3_size=s3_metadata.size,
            s3_last_modified=s3_metadata.last_modified,
        )
