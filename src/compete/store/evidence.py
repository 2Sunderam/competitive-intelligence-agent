from __future__ import annotations

import threading
from pathlib import Path

from compete.models import Claim, EvidenceRecord
from compete.store.validate import quote_is_substring

# One store instance per evidence file, shared by every concurrent extract task.
#
# The map stage fans out one task per document. If each task built its own
# EvidenceStore, each would read the (still empty) file, start its counters at
# zero and hand out ``ev_0001`` / ``cl_0001`` simultaneously — colliding IDs
# that destroy traceability. Sharing one locked instance fixes ID allocation
# and the append race on the JSONL file at once.
_INSTANCES: dict[str, "EvidenceStore"] = {}
_REGISTRY_LOCK = threading.Lock()


def get_evidence_store(path: Path | str) -> "EvidenceStore":
    """Return the shared store for ``path``, creating it once."""
    key = str(Path(path).resolve())
    with _REGISTRY_LOCK:
        store = _INSTANCES.get(key)
        if store is None:
            from compete.config import get_settings

            store = EvidenceStore(
                Path(path), excerpt_chars=get_settings().evidence_excerpt_chars
            )
            _INSTANCES[key] = store
        return store


def reset_evidence_stores() -> None:
    """Drop cached stores. Used between runs and in tests."""
    with _REGISTRY_LOCK:
        _INSTANCES.clear()


class EvidenceStore:
    """Append-only JSONL evidence store with quote validation.

    Prefer :func:`get_evidence_store` over constructing this directly, so all
    concurrent writers share one lock and one ID counter.
    """

    def __init__(self, path: Path, *, excerpt_chars: int = 14000) -> None:
        self.path = path
        self.excerpt_chars = excerpt_chars
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._evidence_count = 0
        self._claim_count = 0
        self._load_index()

    def _load_index(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = EvidenceRecord.model_validate_json(line)
                self._evidence_count += 1
                self._claim_count += len(record.claims)

    def next_evidence_id(self) -> str:
        self._evidence_count += 1
        return f"ev_{self._evidence_count:04d}"

    def next_claim_id(self) -> str:
        self._claim_count += 1
        return f"cl_{self._claim_count:04d}"

    def append(
        self,
        *,
        competitor: str,
        source: str,
        url: str,
        platform: str,
        date: str | None,
        source_text: str,
        draft_claims: list[Claim],
        raw_excerpt: str | None = None,
    ) -> EvidenceRecord | None:
        """Validate quotes and append one evidence line.

        Returns None if no valid claims remain after filtering.
        """
        with self._lock:
            validated: list[Claim] = []
            for draft in draft_claims:
                if not quote_is_substring(draft.quote, source_text):
                    continue
                validated.append(
                    Claim(
                        claim_id=self.next_claim_id(),
                        dimension=draft.dimension,
                        text=draft.text,
                        quote=draft.quote,
                        sentiment=draft.sentiment,
                    )
                )

            if not validated:
                return None

            record = EvidenceRecord(
                evidence_id=self.next_evidence_id(),
                competitor=competitor,
                source=source,  # type: ignore[arg-type]
                url=url,
                platform=platform,
                date=date,
                claims=validated,
                raw_excerpt=(raw_excerpt or source_text)[: self.excerpt_chars],
            )
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")
            return record

    def read_all(self) -> list[EvidenceRecord]:
        if not self.path.exists():
            return []
        records: list[EvidenceRecord] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(EvidenceRecord.model_validate_json(line))
        return records

    def claims_for_competitor(self, competitor: str) -> list[tuple[EvidenceRecord, Claim]]:
        out: list[tuple[EvidenceRecord, Claim]] = []
        for record in self.read_all():
            if record.competitor == competitor:
                for claim in record.claims:
                    out.append((record, claim))
        return out

    def all_claims(self) -> list[tuple[EvidenceRecord, Claim]]:
        out: list[tuple[EvidenceRecord, Claim]] = []
        for record in self.read_all():
            for claim in record.claims:
                out.append((record, claim))
        return out

    def claim_lookup(self) -> dict[str, tuple[EvidenceRecord, Claim]]:
        return {claim.claim_id: (record, claim) for record, claim in self.all_claims()}
