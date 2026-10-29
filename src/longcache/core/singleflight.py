"""Per-instance exact-request coordination; completion objects work across threads/loops."""

from concurrent.futures import Future
from dataclasses import dataclass, field
import threading

from .errors import SingleflightCapacityError, SingleflightInvalidatedError, CacheClosedError


@dataclass(frozen=True)
class Completion:
    kind: str
    payload: bytes | None = None
    error: BaseException | None = field(default=None, repr=False)
    expires_at: float | None = None
    entry_id: int | None = None


@dataclass(eq=False)
class Flight:
    key: tuple = field(repr=False)
    future: Future = field(default_factory=Future, repr=False)
    followers: int = 0
    retained_bytes: int = 0


class Singleflight:
    def __init__(self, config):
        self._lock = threading.Lock()
        self._flights = {}
        self._followers = 0
        self._retained = 0
        self.max_inflight = config.singleflight_max_inflight
        self.max_followers = config.singleflight_max_waiters_per_key
        self.max_payload = config.max_capture_bytes
        self.max_retained = 32 * 1024 * 1024
        self.epoch = 0

    def claim(self, key):
        with self._lock:
            if key[2] != self.epoch:
                raise SingleflightInvalidatedError(
                    "Flight identity was invalidated before registration"
                )
            flight = self._flights.get(key)
            if flight is not None:
                if flight.followers >= self.max_followers:
                    raise SingleflightCapacityError("Singleflight follower capacity reached")
                flight.followers += 1
                self._followers += 1
                return flight, False
            if len(self._flights) >= self.max_inflight:
                raise SingleflightCapacityError("Singleflight key capacity reached")
            flight = Flight(key)
            self._flights[key] = flight
            return flight, True

    def complete(self, flight, completion):
        with self._lock:
            if flight.future.done():
                return
            size = len(completion.payload or b"")
            if size > self.max_payload or size + self._retained > self.max_retained:
                completion, size = Completion("non_shareable"), 0
            if flight.followers:
                flight.retained_bytes = size
                self._retained += size
            if self._flights.get(flight.key) is flight:
                del self._flights[flight.key]
            # Futures are private: callbacks only bridge completion to waiting loops.
            flight.future.set_result(completion)

    def detach(self, flight):
        with self._lock:
            flight.followers -= 1
            self._followers -= 1
            if not flight.followers:
                self._retained -= flight.retained_bytes
                flight.retained_bytes = 0

    def abort(self, shutdown=False):
        error = (
            CacheClosedError("Cache shutdown interrupted shared execution")
            if shutdown
            else SingleflightInvalidatedError("Shared execution was invalidated")
        )
        with self._lock:
            if not shutdown:
                self.epoch += 1
            flights = list(self._flights.values())
            self._flights.clear()
            for flight in flights:
                if not flight.future.done():
                    flight.future.set_result(Completion("invalidated", error=error))

    def snapshot(self):
        with self._lock:
            return {
                "active_flights": len(self._flights),
                "followers": self._followers,
                "retained_replay_bytes": self._retained,
            }
