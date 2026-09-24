"""Which minutes of the trailing week something happened in, for node availability."""

import base64

WINDOW_MINUTES = 7 * 24 * 60
DAY_MINUTES = 24 * 60


def minute_of(t: float) -> int:
    """The epoch minute a wall-clock time falls in."""
    return int(t // 60)


class MinuteRing:
    """A bitmap of the minutes in the trailing week in which something happened.

    Slot i holds epoch minute m where m % WINDOW_MINUTES == i, so two rings
    compare slot for slot. `_head` is the newest minute marked. Advancing it
    clears the slots it passes over, so a slot never holds a minute from
    before the week that ends at the head.
    """

    def __init__(self) -> None:
        self._bits = bytearray(WINDOW_MINUTES // 8)
        self._head: int | None = None

    def _set(self, minute: int) -> None:
        slot = minute % WINDOW_MINUTES
        self._bits[slot >> 3] |= 1 << (slot & 7)

    def _clear(self, minute: int) -> None:
        slot = minute % WINDOW_MINUTES
        self._bits[slot >> 3] &= ~(1 << (slot & 7)) & 0xFF

    def mark(self, minute: int) -> None:
        head = self._head
        if head is not None and minute <= head:
            if minute > head - WINDOW_MINUTES:
                self._set(minute)
            return
        if head is None or minute - head >= WINDOW_MINUTES:
            self._bits = bytearray(len(self._bits))
        else:
            for passed in range(head + 1, minute):
                self._clear(passed)
        self._set(minute)
        self._head = minute

    def bits(self, first: int, last: int) -> int:
        """The marked minutes from first to last inclusive, as a bitmask by slot."""
        if self._head is None:
            return 0
        first = max(first, self._head - WINDOW_MINUTES + 1, last - WINDOW_MINUTES + 1)
        last = min(last, self._head)
        if first > last:
            return 0
        lo, hi = first % WINDOW_MINUTES, last % WINDOW_MINUTES
        if lo <= hi:
            mask = ((1 << (hi - lo + 1)) - 1) << lo
        else:
            mask = (((1 << (WINDOW_MINUTES - lo)) - 1) << lo) | ((1 << (hi + 1)) - 1)
        return int.from_bytes(self._bits, "little") & mask

    def to_state(self) -> dict:
        return {"head": self._head, "bits": base64.b64encode(bytes(self._bits)).decode("ascii")}

    @classmethod
    def from_state(cls, state: dict) -> "MinuteRing":
        """The saved ring, or an empty one if it was saved over another window,
        whose slots would not line up with these."""
        ring = cls()
        bits = base64.b64decode(state["bits"])
        if len(bits) == len(ring._bits):
            ring._head = state["head"]
            ring._bits = bytearray(bits)
        return ring


def share(seen: MinuteRing, up: MinuteRing, first: int, last: int) -> tuple[int, int]:
    """Of the minutes from first to last in which `up` is marked, how many `seen` marks too."""
    up_bits = up.bits(first, last)
    return (seen.bits(first, last) & up_bits).bit_count(), up_bits.bit_count()
