"""Explicit mapping from solution-stream identity to physical evidence source."""

from __future__ import annotations

import json
from pathlib import Path


class ProvenanceRegistry:
    def __init__(self, stream_to_source: dict[str, str]) -> None:
        if not stream_to_source:
            raise ValueError("provenance registry must not be empty")
        if any(not key or not value for key, value in stream_to_source.items()):
            raise ValueError("empty stream or physical-source identifier")
        self._stream_to_source = dict(stream_to_source)

    @classmethod
    def load(cls, path: Path) -> "ProvenanceRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            {
                str(stream): str(source)
                for stream, source in payload[
                    "solution_stream_to_physical_source"
                ].items()
            }
        )

    @staticmethod
    def solution_stream_id(path: Path) -> str:
        name = Path(path).name
        suffix = ".estimator_input.npz"
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        fields = name.split("__")
        if len(fields) < 3:
            raise ValueError(f"missing explicit solution-stream field: {path}")
        return fields[1]

    def physical_source(self, path: Path) -> str:
        stream = self.solution_stream_id(path)
        try:
            return self._stream_to_source[stream]
        except KeyError as error:
            raise ValueError(
                f"solution stream {stream!r} is absent from provenance registry"
            ) from error
