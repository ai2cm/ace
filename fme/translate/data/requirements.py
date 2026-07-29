"""What the objectives need from the data streams, and how it merges.

This module is the seam between the objectives (which know what they consume)
and the data layer (which knows how to load it). Each objective declares an
:class:`ObjectiveDataRequirements` — for every stream it consumes, the variables
it reads and how long a time window it needs.
:meth:`TranslateDataRequirements.from_objectives` merges the whole objective
list into

- one :class:`~fme.ace.requirements.DataRequirements` per stream, and
- the *pairing groups*: which streams must be sampled at the same valid times.

Pairing is derived rather than configured. Every objective ties together the
streams it consumes (a translation objective comparing 1° and 2° fields of the
same state is meaningless if the two are drawn from different times), so the
co-occurrence graph's connected components are exactly the sets that must share
a time index. Streams that never co-occur are sampled independently, which is
what the transfer-learning program's ERA5 and SHiELD streams want.

Nothing here depends on the component pool: the loader takes its variable names
from these requirements, not from the pool's domain channel lists.
"""

import dataclasses
from collections.abc import Mapping, Sequence

from fme.ace.requirements import DataRequirements
from fme.core.dataset.schedule import IntMilestone, IntSchedule

__all__ = [
    "ObjectiveDataRequirements",
    "StreamRequirements",
    "TranslateDataRequirements",
]


@dataclasses.dataclass(frozen=True)
class StreamRequirements:
    """One objective's need from one data stream.

    Parameters:
        names: Names of the variables the objective reads from this stream.
        n_timesteps: Number of timesteps per sample window, including the
            initial condition(s). May be an :class:`IntSchedule` for an
            objective whose rollout length grows over epochs.
    """

    names: list[str]
    n_timesteps: int | IntSchedule = 1

    def __post_init__(self):
        if not self.names:
            raise ValueError("StreamRequirements requires at least one variable name.")

    @property
    def n_timesteps_schedule(self) -> IntSchedule:
        # The int-or-schedule union is the shape a config author writes (a plain
        # int being the common case), so narrowing it needs this check; the same
        # one ace's DataRequirements.n_timesteps_schedule makes.
        if isinstance(self.n_timesteps, IntSchedule):
            return self.n_timesteps
        return IntSchedule.from_constant(self.n_timesteps)


@dataclasses.dataclass(frozen=True)
class ObjectiveDataRequirements:
    """One objective's needs, keyed by stream name.

    Parameters:
        streams: What this objective needs from each stream it consumes. Every
            stream named here is tied to every other one: they will be sampled
            at the same valid times.
    """

    streams: Mapping[str, StreamRequirements]

    def __post_init__(self):
        if not self.streams:
            raise ValueError("An objective must consume at least one data stream.")


def _ordered_union(name_lists: Sequence[Sequence[str]]) -> list[str]:
    """The union of ``name_lists`` in first-seen order."""
    result: list[str] = []
    seen: set[str] = set()
    for names in name_lists:
        for name in names:
            if name not in seen:
                seen.add(name)
                result.append(name)
    return result


def _pointwise_max(schedules: Sequence[IntSchedule]) -> IntSchedule:
    """The schedule taking the maximum of ``schedules`` at every epoch.

    A max (rather than an error on disagreement) is the correct merge because
    ace loads a window of the scheduled length and each objective samples a
    prefix of it: loading the longest window any objective asks for serves them
    all. The merged schedule's milestones are the union of the inputs'.
    """
    epochs = sorted({milestone.epoch for s in schedules for milestone in s.milestones})
    start_value = max(s.get_value(0) for s in schedules)
    milestones: list[IntMilestone] = []
    previous = start_value
    for epoch in epochs:
        value = max(s.get_value(epoch) for s in schedules)
        if value != previous:
            milestones.append(IntMilestone(epoch=epoch, value=value))
            previous = value
    return IntSchedule(start_value=start_value, milestones=milestones)


def _pairing_groups(
    objectives: Sequence[ObjectiveDataRequirements],
) -> list[list[str]]:
    """Connected components of the stream co-occurrence graph.

    Each objective connects the streams it consumes; the components are the
    groups sampled at a shared time index. Grouping is transitive: objectives
    (a, b) and (b, c) put all three in one group. A stream no multi-stream
    objective touches is its own singleton group.

    Groups, and the streams within each group, are ordered by first appearance
    in ``objectives`` so the result is deterministic.
    """
    parent: dict[str, str] = {}
    first_seen: list[str] = []

    def find(name: str) -> str:
        root = name
        while parent[root] != root:
            root = parent[root]
        while parent[name] != root:
            parent[name], name = root, parent[name]
        return root

    for objective in objectives:
        names = list(objective.streams)
        for name in names:
            if name not in parent:
                parent[name] = name
                first_seen.append(name)
        for other in names[1:]:
            parent[find(other)] = find(names[0])

    groups: dict[str, list[str]] = {}
    for name in first_seen:
        groups.setdefault(find(name), []).append(name)
    return list(groups.values())


@dataclasses.dataclass(frozen=True)
class TranslateDataRequirements:
    """The whole objective list's needs, as the data layer consumes them.

    Parameters:
        streams: Merged requirements for each stream any objective consumes.
        pairing_groups: Partition of ``streams`` into sets sampled at the same
            valid times. Each group is sampled independently of the others.
    """

    streams: Mapping[str, DataRequirements]
    pairing_groups: Sequence[Sequence[str]]

    def __post_init__(self):
        grouped = [name for group in self.pairing_groups for name in group]
        # Checked before the partition check, which would otherwise report a
        # duplicated stream as a partition mismatch.
        if len(set(grouped)) != len(grouped):
            raise ValueError(
                f"A stream may appear in only one pairing group; got {grouped}."
            )
        if sorted(grouped) != sorted(self.streams):
            raise ValueError(
                "pairing_groups must partition the streams; got groups over "
                f"{sorted(grouped)} for streams {sorted(self.streams)}."
            )
        missing_allowed = sorted(
            name
            for name, requirements in self.streams.items()
            if requirements.allow_missing_variables
        )
        if missing_allowed:
            # The loader passes no allow_missing_variables through to
            # DatasetABC.build, so honoring it would need plumbing in
            # StreamConfig.get_dataset; refuse rather than silently load
            # every variable.
            raise ValueError(
                "allow_missing_variables is not supported by the translate data "
                f"loader, but is set on streams {missing_allowed}."
            )

    @classmethod
    def from_objectives(
        cls, objectives: Sequence[ObjectiveDataRequirements]
    ) -> "TranslateDataRequirements":
        """Merge each objective's needs into per-stream requirements and groups.

        Per stream, the variable names are the ordered union of what the
        objectives read and the window length is the pointwise maximum of their
        schedules. The pairing groups come from objective co-occurrence.
        """
        if not objectives:
            raise ValueError("At least one objective is required to load data.")
        names: dict[str, list[Sequence[str]]] = {}
        schedules: dict[str, list[IntSchedule]] = {}
        for objective in objectives:
            for stream_name, stream in objective.streams.items():
                names.setdefault(stream_name, []).append(stream.names)
                schedules.setdefault(stream_name, []).append(
                    stream.n_timesteps_schedule
                )
        streams = {
            stream_name: DataRequirements(
                names=_ordered_union(names[stream_name]),
                n_timesteps=_pointwise_max(schedules[stream_name]),
            )
            for stream_name in names
        }
        return cls(streams=streams, pairing_groups=_pairing_groups(objectives))
