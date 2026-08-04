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
``work.wait(timedelta)``, which raises on expiry. `timeout_contract_verified` is
what stops a torch upgrade removing that bound silently: on a release the
behaviour was not read against, the caller degrades to the group's own timeout
rather than pretending to a bound it may not have.

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
import logging
import weakref

import torch
import torch.distributed

from fme.core.device import get_device

logger = logging.getLogger(__name__)

# The agreement group's own timeout, matching the 30 minutes
# `torch_distributed.py` passes `init_process_group`. Creation is therefore
# exactly as safe as `init_process_group` itself. It is *not* what a rank with
# nothing pending of its own passes as an operation deadline -- that comes from
# `default_group_timeout`, because the default group does not always carry 30
# minutes.
GROUP_TIMEOUT: datetime.timedelta = datetime.timedelta(minutes=30)

# gloo reports a deadline expiry and a peer's socket closing as the same
# `RuntimeError` type, so the message is the only thing that separates them.
#
# A substring match is the whole of that discrimination, and it fails silently in
# both directions: a reworded expiry message would be read as a crash and
# escalated, and a crash whose message happened to contain this would be read as a
# wedge and given up on quietly. `_VERIFIED_TORCH` is what makes the first
# direction safe on the releases the design was checked against, and it is the
# thing to extend if a release ever reworks the message. The second direction is
# not guarded at all and was not introduced here.
_TIMEOUT_MESSAGE: str = "Operation timed out!"

# The torch releases whose private timeout behaviours have been read and, where
# possible, measured: 2.7 is the development environment, 2.8 is what
# `constraints.txt` pins for the Docker image, which is what makes the verified
# path the normal one. Compared as ``(major, minor)`` so a patch bump does not
# change the answer. Deliberately not widened to releases nobody checked -- the
# point of `timeout_contract_verified` is that an unverified release degrades to
# the group's own timeout rather than pretending to a bound it may not have.
_VERIFIED_TORCH: frozenset[tuple[int, int]] = frozenset({(2, 7), (2, 8)})

# `warn_if_timeout_contract_unverified` writes once per process, not once per
# group: a job builds one agreement group, but a test session builds many.
_warned_unverified: bool = False


def timeout_contract_verified() -> bool:
    """Whether the installed torch is one whose timeout behaviours were checked.

    Three behaviours the design's short bound rests on are private torch internals
    read out of headers rather than documented API: ``work.wait(timedelta)``
    raises rather than returning ``False`` on expiry; ``kNoTimeout`` is
    ``milliseconds(0)``, so a *zero* deadline means *no* deadline; and the pybind
    chrono caster truncates a ``timedelta`` to integral milliseconds, so a
    sub-millisecond deadline becomes that same sentinel. None can be probed
    without a collective made to time out, so what is checked here is the release
    they were verified against.

    ``_set_pg_timeout`` is the cautionary tale: it exists, is documented, and
    explicitly handles ``ProcessGroupGloo`` -- and does nothing whatever to an
    already-constructed gloo context. An upgrade that quietly changed any of the
    three above would remove the bound just as silently.

    **``False`` degrades the mechanism; it never fails a job**, and both other
    options were tried and are worse. `pyproject.toml` declares ``torch>=2.4.0``,
    so unverified releases are supported releases, and every distributed job
    builds an agreement group -- including inference and evaluator runs that never
    stop cooperatively. Raising at construction breaks all of those: on torch
    2.6.0, which the multi-rank CPU matrix runs, it errored all 80 parallel tests
    at setup on all eight configurations. Raising at first use instead moves the
    failure into a preemption, losing the job at the one moment the teardown
    matters. So a rank on an unverified release agrees with its peers exactly as
    usual and passes the default group's own timeout where it would have passed a
    short one. That is `main`'s behaviour, not a new failure mode: the ranks still
    leave together, and only the short bound on a rank *giving up* is unavailable.
    """
    major, minor = torch.__version__.split(".")[:2]
    return (int(major), int(minor)) in _VERIFIED_TORCH


def warn_if_timeout_contract_unverified() -> None:
    """Say once per process, where the group is built, that the bound is degraded.

    ``logging`` rather than `write_marker`: keeping this module free of any
    dependency on the signal-handling half is what keeps every claim about the
    collective testable multi-rank. `fme/core/logging_utils.py` puts non-root
    ranks at ERROR, so in a real job this is root's line only -- which is enough
    for a human, because the version is the same on every rank. The per-rank
    machine-readable record is `cooperative_stop`'s ``agreement-bound`` marker,
    emitted where the degraded bound is actually applied.
    """
    global _warned_unverified
    if _warned_unverified or timeout_contract_verified():
        return
    _warned_unverified = True
    verified = ", ".join(f"{major}.{minor}" for major, minor in sorted(_VERIFIED_TORCH))
    logger.warning(
        "torch %s has not been checked against the operation-timeout behaviours "
        "the cooperative stop's short deadline relies on (verified: %s). Ranks "
        "will still agree on a batch boundary to leave the loop at, but a rank "
        "that gives up waiting for a peer is bounded only by the default group's "
        "own timeout rather than by a few seconds. To restore the short bound, "
        "re-read `work.wait`'s raise on expiry, `kNoTimeout` in `Work.hpp`, and "
        "the pybind chrono caster's millisecond truncation, then add this release "
        "to `_VERIFIED_TORCH`.",
        torch.__version__,
        verified,
    )


def default_group_timeout() -> float:
    """Seconds the *default* group -- the gradient all-reduce's -- will wait.

    Read off the group rather than restated, because the two backends configure it
    differently and a restated figure was wrong for one of them:
    `torch_distributed.py` passes `GROUP_TIMEOUT` to ``init_process_group``
    explicitly, while `ModelTorchDistributed` initialises through
    ``DistributedManager``, which passes none -- so on GPU that group carries
    NCCL's default 10 minutes, a third of the figure a hardcoded 30 minutes would
    claim to mirror.

    Falls back to `GROUP_TIMEOUT` where there is no default group to read, which
    is every single-rank run and every test that builds a `CooperativeStop`
    without a communicator; there is no gradient all-reduce there to mirror, and
    the agreement is `SoloStopAgreement`, which never blocks.

    Falls back to it and says so if torch has moved the private attributes this
    reads. Failing instead would break a training loop at entry over a diagnostic
    figure, and the fallback is the value this was before it was derived -- so the
    consequence is that the mirroring claim goes back to being asserted rather than
    true, which the warning is there to report.
    """
    if not torch.distributed.is_initialized():
        return GROUP_TIMEOUT.total_seconds()
    try:
        group = torch.distributed.distributed_c10d._get_default_group()
        # the device this process computes on is the device the gradient all-reduce
        # runs on, so it selects the backend whose deadline is the one to mirror
        backend = group._get_backend(get_device())
        timeout: datetime.timedelta = backend.options._timeout
    except (AttributeError, RuntimeError, KeyError):
        logger.warning(
            "Could not read the default process group's own timeout, so the "
            "cooperative stop will use %s for a rank with nothing pending of its "
            "own. That figure is only correct where the default group carries it; "
            "under a backend that took torch's own default it is too loose.",
            GROUP_TIMEOUT,
            exc_info=True,
        )
        return GROUP_TIMEOUT.total_seconds()
    return timeout.total_seconds()


def _check_deadline(timeout: float) -> None:
    """Reject a deadline torch would read as *no* deadline.

    On every implementation, including `SoloStopAgreement`, so that the contract
    `StopAgreement.exchange` documents holds for all of them rather than only the
    one that blocks.
    """
    if timeout <= 0.0:
        raise ValueError(
            f"a stop agreement deadline must be positive, got {timeout}: "
            "torch reads a zero-length timeout as *no* timeout, so this "
            "would wait for the group's own timeout instead"
        )


def is_deadline_expiry(err: BaseException) -> bool:
    """Whether ``work.wait`` raised because its own deadline expired.

    The distinction is load-bearing: a deadline expiry means a peer is wedged and
    this rank is giving up on it, while ``Connection closed by peer`` means the
    peer's process has died and the caller should let that surface as the crash it
    is rather than as a graceful stop. gloo raises the same ``RuntimeError`` type
    for both, so only the message tells them apart -- which is why
    `_VERIFIED_TORCH` pins the releases that message was read from.
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
                so passing it would wait for the group's own timeout. Checked on
                every implementation, `SoloStopAgreement` included, so this
                contract holds for all of them and not only for the one that
                blocks.

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
        _check_deadline(timeout)
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
    ``| None`` check. It never blocks, and the only thing it can raise is the
    deadline check every implementation owes the abstract contract -- which fires
    on a caller bug rather than on anything a peer did.
    """

    @property
    def world_size(self) -> int:
        return 1

    @property
    def abandoned(self) -> bool:
        return False

    def exchange(self, reason: int, index: int, timeout: float) -> tuple[int, int, int]:
        # checked despite never waiting, so a deadline this implementation would
        # have ignored is still caught on the single-rank runs that exercise it
        _check_deadline(timeout)
        return reason, index, index


# Groups that must never be collected, because dropping the last reference to a
# gloo group with outstanding work blocks in `~ProcessGroupGloo()` without bound.
# A superseded group is moved here rather than released.
_leaked: list[StopAgreement] = []

_agreement: StopAgreement | None = None

# A weak reference to the default group `_agreement` was built alongside, which is
# what tells a cached group from one belonging to a world that has been destroyed
# and re-initialised. Weak deliberately: a strong reference here would make this
# module the potential last holder of the *default* group, whose destructor carries
# the same unbounded wait, and a collected referent reads as "not the current
# world", which is the answer we want anyway.
_agreement_world: "weakref.ReferenceType[torch.distributed.ProcessGroup] | None" = None


def build_stop_agreement(world_size: int) -> GlooStopAgreement:
    """Create a new agreement group. Uncached, and a collective on every rank.

    ``new_group`` is itself a collective, so every rank must call this, and in
    the same order relative to any other ``new_group``. The only production
    caller is `new_stop_agreement`; the other callers are tests that want a
    throwaway group of their own.

    Separate from `new_stop_agreement` because a cache and a throwaway test group
    cannot share one entry point: a cached accessor would hand the first test the
    process-wide group -- the session's own, which must never be the one a test
    times out -- and every later test whatever that test left behind.

    Args:
        world_size: Ranks in the group. Not ``ranks``, which is ``new_group``'s
            own parameter for a *list* of ranks; this group spans them all.
    """
    # here, and not where a deadline is used, because this is the one point every
    # job reaches exactly once. It warns and returns; see
    # `timeout_contract_verified` for why it must not raise.
    warn_if_timeout_contract_unverified()
    group = torch.distributed.new_group(backend="gloo", timeout=GROUP_TIMEOUT)
    return GlooStopAgreement(group, world_size)


def new_stop_agreement(
    world_size: int, world: torch.distributed.ProcessGroup
) -> StopAgreement:
    """The process-wide agreement group, created once per world. What backends call.

    A second call against the *same* world returns the same group rather than
    issuing a second ``new_group`` some ranks may not match -- which a backend
    constructed twice in one process really does reach, since
    ``DistributedManager.cleanup()`` wipes the state that would otherwise stop it.

    The invalidation predicate is the identity of the default group, because that
    is the thing that actually changes: a cached group belongs to the world it was
    created in, and after a ``destroy_process_group()`` and a fresh
    ``init_process_group()`` its gloo context is no part of the new world's
    bookkeeping. ``is_initialized()`` cannot answer this -- every production caller
    runs immediately after ``init_process_group``, so it is always ``True`` there.

    A superseded group is **moved, not released**: dropping the last reference to
    a gloo group runs ``~ProcessGroupGloo()``, which waits without bound for a
    work queue an abandoned exchange may have left non-empty.

    Args:
        world_size: Ranks in the group, which spans the whole world.
        world: The default group this agreement will belong to. Passed rather than
            looked up so the caller's own ``_get_default_group()`` -- which every
            caller already makes, to hold for the teardown watchdog -- is the one
            identity in play.
    """
    global _agreement, _agreement_world
    if _agreement is not None:
        cached = None if _agreement_world is None else _agreement_world()
        if cached is world:
            return _agreement
        # never `_agreement = None`: that is the unbounded block
        _leaked.append(_agreement)
    _agreement = build_stop_agreement(world_size)
    _agreement_world = weakref.ref(world)
    return _agreement
