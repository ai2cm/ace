"""Agree across ranks, in bounded time, that a loop is being left.

A rank that decides on its own to leave a batch loop strands its peers: they are
inside a gradient all-reduce nobody will complete, and a stranded rank can join
no later collective, so no subsequent agreement can rescue it. The only way two
ranks can leave at the same iteration is to communicate before either commits to
entering the next one.

This module owns that communication and nothing else. One
``all_reduce(MAX)`` of a three-element CPU ``int64`` vector over a dedicated,
world-wide gloo group, with a deadline the caller chooses per operation. It has
no signal handling and no notion of *why* a rank is leaving, which is what keeps
every claim about the collective testable multi-rank without sending a signal
into a live pytest session.

Two properties are load-bearing and neither is obvious.

**The deadline is per operation, not per group.** A gloo group can neither be
created with a short timeout -- creation then fails on every rank, asymmetrically,
on launch jitter alone -- nor shortened afterwards, ``_set_pg_timeout`` being
silently inert on an already-constructed gloo context. What does work is
``work.wait(timedelta)``, which raises on expiry. `assert_timeout_contract` is
what stops a torch upgrade removing that bound silently.

**The group is never reclaimed.** ``~ProcessGroupGloo()`` waits without bound for
a work queue that an abandoned exchange leaves non-empty, and nothing else drains
it: gloo overrides neither ``shutdown()`` nor ``abort()``. So a group that has
been given up on is *held* for the life of the process -- which is also what makes
``destroy_process_group()`` return, since it only clears torch's own bookkeeping.
The design's bound after an abandoned exchange is the process exiting, not the
group coming back.
"""

import abc
import datetime

import torch
import torch.distributed

# The agreement group's own timeout, matching the 30 minutes
# `torch_distributed.py` gives the default group. Creation is therefore exactly
# as safe as `init_process_group` itself, and a rank with nothing pending of its
# own can pass this as its operation deadline without inventing a second number.
GROUP_TIMEOUT: datetime.timedelta = datetime.timedelta(minutes=30)

# gloo reports a deadline expiry and a peer's socket closing as the same
# `RuntimeError` type, so the message is the only thing that separates them.
_TIMEOUT_MESSAGE: str = "Operation timed out!"

# The torch releases whose private timeout behaviours have been read and, where
# possible, measured: 2.7 is the development environment, 2.8 is what
# `constraints.txt` pins for the Docker image. Compared as ``(major, minor)`` so a
# patch bump does not fail a job. A release outside this set has not been checked
# and may have moved a behaviour the design's only bound rests on.
_VERIFIED_TORCH: frozenset[tuple[int, int]] = frozenset({(2, 7), (2, 8)})


def assert_timeout_contract() -> None:
    """Fail at group creation if torch's operation-timeout contract may have moved.

    Three behaviours this module's only bound rests on are private torch
    internals read out of headers rather than documented API:
    ``work.wait(timedelta)`` raises rather than returning ``False`` on expiry;
    ``kNoTimeout`` is ``milliseconds(0)``, so a *zero* deadline means *no*
    deadline; and the pybind chrono caster truncates a ``timedelta`` to integral
    milliseconds, so a sub-millisecond deadline becomes that same sentinel. None
    can be probed without a collective made to time out, so what is checked here
    is the release they were verified against.

    ``_set_pg_timeout`` is the cautionary tale: it exists, is documented, and
    explicitly handles ``ProcessGroupGloo`` -- and does nothing whatever to an
    already-constructed gloo context. An upgrade that quietly changed any of the
    three above would remove the bound just as silently, so this fails loudly
    instead.

    Raises:
        RuntimeError: If the installed torch is not one of the verified releases.
    """
    major, minor = torch.__version__.split(".")[:2]
    installed = (int(major), int(minor))
    if installed not in _VERIFIED_TORCH:
        verified = ", ".join(
            f"{major}.{minor}" for major, minor in sorted(_VERIFIED_TORCH)
        )
        raise RuntimeError(
            f"torch {torch.__version__} has not been checked against the "
            "operation-timeout behaviours the cooperative stop relies on "
            f"(verified: {verified}). Re-read `work.wait`'s raise on expiry, "
            "`kNoTimeout` in `Work.hpp`, and the pybind chrono caster's "
            "millisecond truncation, then add this release to "
            "`_VERIFIED_TORCH`."
        )


def is_deadline_expiry(err: BaseException) -> bool:
    """Whether ``work.wait`` raised because its own deadline expired.

    The distinction is load-bearing: a deadline expiry means a peer is wedged and
    this rank is giving up on it, while ``Connection closed by peer`` means the
    peer's process has died and the caller should let that surface as the crash it
    is rather than as a graceful stop. gloo raises the same ``RuntimeError`` type
    for both, so only the message tells them apart -- which is why
    `assert_timeout_contract` pins the release that message was read from.
    """
    return _TIMEOUT_MESSAGE in str(err)


class StopAgreement(abc.ABC):
    """One deadline-bounded exchange over a group every rank has joined."""

    @property
    @abc.abstractmethod
    def world_size(self) -> int:
        """Ranks taking part, carried so the evidence line can report it."""

    @property
    @abc.abstractmethod
    def abandoned(self) -> bool:
        """Whether an exchange was given up on, leaving work outstanding.

        Once this is true the group can never be reclaimed in bounded time, so
        the caller owes an ``agreement-abandoned`` evidence line and no code path
        may drop the group's reference. Always ``False`` on `SoloStopAgreement`,
        which never blocks.
        """

    @abc.abstractmethod
    def exchange(self, reason: int, index: int, timeout: float) -> tuple[int, int, int]:
        """Reduce ``[reason, +index, -index]`` with ``MAX``.

        Args:
            reason: A rank's reason code for leaving, ``0`` meaning it is not.
                ``MAX`` over these is both a logical OR and a choice of the more
                serious reason.
            index: This rank's iteration index. Carried twice, negated once, so
                the reduction yields both the highest and the lowest index any
                rank contributed.
            timeout: Seconds this rank will wait. Always strictly positive: zero
                is torch's *no-timeout* sentinel rather than an immediate expiry,
                so passing it would wait for the group's 30 minutes.

        Returns:
            ``(reason, max_index, min_index)``, the negation on the third element
            already undone, so the caller's symmetry check is plain equality.

        Raises:
            RuntimeError: On expiry of ``timeout``, or when a peer's socket
                closes. `is_deadline_expiry` separates the two.
            ValueError: If ``timeout`` is not strictly positive.
        """


class GlooStopAgreement(StopAgreement):
    """A real exchange, over a gloo group this object holds for good.

    The group reference is what makes ``destroy_process_group()`` return: that
    call clears torch's bookkeeping but drains nothing, and the gloo destructor --
    the only thing that does drain the work queue, and unbounded -- runs only when
    the last reference goes.
    """

    def __init__(self, group: torch.distributed.ProcessGroup, world_size: int) -> None:
        self._group = group
        self._world_size = world_size
        self._abandoned = False

    @property
    def group(self) -> torch.distributed.ProcessGroup:
        """The group itself, for a caller that needs to name it to torch.

        Read it, do not keep it: this object's reference is what stops the group
        being collected, and a caller that outlives it while holding the last
        reference would run the unbounded destructor.
        """
        return self._group

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def abandoned(self) -> bool:
        return self._abandoned

    def exchange(self, reason: int, index: int, timeout: float) -> tuple[int, int, int]:
        if timeout <= 0.0:
            raise ValueError(
                f"a stop agreement deadline must be positive, got {timeout}: "
                "torch reads a zero-length timeout as *no* timeout, so this "
                "would wait for the group's own 30 minutes"
            )
        payload = torch.tensor([reason, index, -index], dtype=torch.int64)
        work = torch.distributed.all_reduce(
            payload,
            op=torch.distributed.ReduceOp.MAX,
            group=self._group,
            async_op=True,
        )
        try:
            work.wait(datetime.timedelta(seconds=timeout))
        except BaseException:
            # The operation stays in gloo's work queue, so this group is now
            # unreclaimable in bounded time and the caller has to say so.
            self._abandoned = True
            raise
        reduced = payload.tolist()
        return int(reduced[0]), int(reduced[1]), -int(reduced[2])


class SoloStopAgreement(StopAgreement):
    """One rank, or a DataLoader worker: no group, no collective.

    `exchange` returns the caller's own values unreduced, so ``max_index ==
    min_index`` holds trivially and no call site needs an ``isinstance`` or a
    ``| None`` check. It never blocks and never raises.
    """

    @property
    def world_size(self) -> int:
        return 1

    @property
    def abandoned(self) -> bool:
        return False

    def exchange(self, reason: int, index: int, timeout: float) -> tuple[int, int, int]:
        return reason, index, index


# Groups that must never be collected, because dropping the last reference to a
# gloo group with outstanding work blocks in `~ProcessGroupGloo()` without bound.
# A superseded group is moved here rather than released.
_leaked: list[StopAgreement] = []

_agreement: StopAgreement | None = None


def build_stop_agreement(world_size: int) -> GlooStopAgreement:
    """Create a new agreement group. Uncached, and a collective on every rank.

    ``new_group`` is itself a collective, so every rank must call this, and in
    the same order relative to any other ``new_group``. The only production
    caller is `new_stop_agreement`; the other callers are tests that want a
    throwaway group of their own.

    Separate from `new_stop_agreement` because a cache and a throwaway test group
    cannot share one entry point: ``is_initialized()`` does not flip when a test
    destroys a single group, so a cached accessor would hand the first test the
    process-wide group -- the session's own -- and every later test a destroyed
    one.

    Args:
        world_size: Ranks in the group. Not ``ranks``, which is ``new_group``'s
            own parameter for a *list* of ranks; this group spans them all.
    """
    assert_timeout_contract()
    group = torch.distributed.new_group(backend="gloo", timeout=GROUP_TIMEOUT)
    return GlooStopAgreement(group, world_size)


def new_stop_agreement(world_size: int) -> StopAgreement:
    """The process-wide agreement group, created once. What backends call.

    A second call in the same process returns the same group rather than issuing
    a second ``new_group`` some ranks may not match -- which a backend
    constructed twice in one process really does reach, since
    ``DistributedManager.cleanup()`` wipes the state that would otherwise stop
    it. ``is_initialized()`` is the invalidation predicate because
    ``destroy_process_group()`` is what flips it, and it clears the bookkeeping
    that would let the group be reused.

    A superseded group is **moved, not released**: dropping the last reference to
    a gloo group runs ``~ProcessGroupGloo()``, which waits without bound for a
    work queue an abandoned exchange may have left non-empty.
    """
    global _agreement
    if _agreement is not None:
        if torch.distributed.is_initialized():
            return _agreement
        # never `_agreement = None`: that is the unbounded block
        _leaked.append(_agreement)
    _agreement = build_stop_agreement(world_size)
    return _agreement
