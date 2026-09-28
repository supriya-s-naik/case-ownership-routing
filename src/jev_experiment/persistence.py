"""Result stores used by the graph persistence node."""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from jev_experiment.models import CaseResult


class ResultStore(Protocol):
    async def save(self, result: CaseResult) -> str | None: ...


@dataclass
class InMemoryResultStore:
    results: list[CaseResult] = field(default_factory=list)

    async def save(self, result: CaseResult) -> str | None:
        self.results.append(result)
        return None


@dataclass
class JsonResultStore:
    """Persist a run as an atomically replaced JSON array."""

    path: Path
    results: list[CaseResult] = field(default_factory=list, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def save(self, result: CaseResult) -> str:
        async with self._lock:
            self.results.append(result)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            temporary_path.write_text(
                json.dumps(
                    [item.model_dump(mode="json") for item in self.results],
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary_path.replace(self.path)
        return str(self.path)
