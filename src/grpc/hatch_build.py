import importlib
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

sys.path.insert(0, str(Path(__file__).parent))
compile_protos = importlib.import_module("scripts.compile_protos").compile_protos


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        compile_protos(self.root)
