"""Row growth, joins and endpoint states (PROJECT.md §10, steps 3-7).

Every row starts from a chain. In rounds, each active endpoint takes one step
of growth_step_m:

1. Candidate directions: the outward tangent turned by every whole degree up
   to max_turn (from the minimum curvature radius: step / radius radians).
2. Each candidate is scored with the five-line template centred on the new
   step, at the row's spacing. The best score wins; ties go to the smaller turn.
3. Supported (best >= growth_min_score): take the step. Otherwise bridge:
   continue straight along the row's direction over the last bridge_tangent_m,
   unsupported. After more than max_gap_m of unsupported steps the
   unsupported tail is removed and the end stops as "unsupported". Every
   segment records whether it was supported, so bridged length is reported.
4. Join: another row's free end that lies at most link_lateral_frac * d to
   the side of the growth line, and between join_distance_frac * d behind the
   tip and that far beyond the step, facing back within join_angle_deg, is
   joined and the two rows merge.
5. Conflict: the step crosses another row, or lands on another row's line
   (closer than link_lateral_frac * d), so the end stops as "conflict".
6. Boundary: the step leaves the block polygon. It is cut at the boundary and
   the end stops as "boundary".
7. Limit: the end has grown max_row_length_m.
Growth ends when no active endpoints remain, so it always terminates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from ..config.settings import ReconstructionSettings
from ..geo import axial, heading

FREE_STATES = ("active", "unsupported")


@dataclass
class Endpoint:
    state: str = "active"
    gap_m: float = 0.0
    tail: int = 0  # trailing unsupported vertices
    grown_m: float = 0.0
    steps: int = 0
    supported_steps: int = 0


@dataclass
class Row:
    id: int
    block_id: int
    points: list[np.ndarray]  # ordered start -> end
    spacing_m: float
    spacing_index: int
    seed_ids: list[int]
    chain_ids: list[int]
    start: Endpoint = field(default_factory=Endpoint)
    end: Endpoint = field(default_factory=Endpoint)
    joined_rows: list[int] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    alive: bool = True
    # seg_support[i]: segment points[i] -> points[i + 1] had image support (False = bridged).
    seg_support: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.seg_support:
            self.seg_support = [True] * max(0, len(self.points) - 1)

    def endpoint(self, side: str) -> Endpoint:
        return self.start if side == "start" else self.end

    def tip(self, side: str) -> np.ndarray:
        return self.points[0] if side == "start" else self.points[-1]

    def outward_tangent(self, side: str, back_m: float) -> np.ndarray:
        """Unit vector pointing out of the row at this end, from a point up to back_m behind the tip."""
        pts = self.points if side == "end" else self.points[::-1]
        tip = pts[-1]
        travelled = 0.0
        ref = pts[-2]
        for k in range(len(pts) - 2, -1, -1):
            ref = pts[k]
            travelled = float(np.hypot(*(tip - ref)))
            if travelled >= back_m:
                break
        v = tip - ref
        norm = float(np.hypot(*v))
        return v / norm if norm > 0 else np.array([1.0, 0.0])

    def line(self) -> LineString:
        return LineString(np.array(self.points))

    def bridged_segments(self) -> list[LineString]:
        """Segments added without image support."""
        return [
            LineString([self.points[i], self.points[i + 1]])
            for i, ok in enumerate(self.seg_support)
            if not ok and np.any(self.points[i] != self.points[i + 1])
        ]

    def reverse(self) -> None:
        self.points.reverse()
        self.seg_support.reverse()
        self.start, self.end = self.end, self.start


@dataclass
class JoinRecord:
    row_id: int
    merged_row_id: int
    x: float
    y: float


class Grower:
    def __init__(self, rows: list[Row], polygon, scorer, settings: ReconstructionSettings, line_length_m: float):
        self.rows = rows
        self.by_id = {r.id: r for r in rows}
        self.polygon = polygon
        shapely.prepare(polygon)
        self.boundary = polygon.boundary
        self.scorer = scorer
        self.s = settings
        self.back_m = max(settings.growth_step_m, line_length_m)
        self.bridge_back_m = max(self.back_m, settings.bridge_tangent_m)
        self.max_turn = int(math.floor(settings.max_turn_deg + 1e-9))
        self.joins: list[JoinRecord] = []
        self.rounds = 0

    def run(self) -> list[Row]:
        while True:
            active = [(r, side) for r in self.rows if r.alive for side in ("start", "end") if r.endpoint(side).state == "active"]
            if not active:
                break
            self._index()
            for row, side in active:
                if row.alive and row.endpoint(side).state == "active":
                    self._step(row, side)
            self.rounds += 1
        return [r for r in self.rows if r.alive]

    # -- spatial indexes, rebuilt every round -------------------------------------------------

    def _index(self) -> None:
        alive = [r for r in self.rows if r.alive]
        self._line_rows = alive
        self._line_tree = shapely.STRtree([r.line() for r in alive])
        ends = [(r, side) for r in alive for side in ("start", "end") if r.endpoint(side).state in FREE_STATES]
        self._ends = ends
        self._end_tree = cKDTree(np.array([r.tip(s) for r, s in ends])) if ends else None

    # -- one growth step ----------------------------------------------------------------------

    def _step(self, row: Row, side: str) -> None:
        s = self.s
        ep = row.endpoint(side)
        P = row.tip(side)
        if not self.polygon.intersects(Point(P)):
            ep.state = "boundary"
            return
        phi = heading(row.outward_tangent(side, self.back_m))
        turns = np.arange(-self.max_turn, self.max_turn + 1, dtype=float)
        turns = turns[np.argsort(np.abs(turns), kind="stable")]  # straight first, for tie-breaking
        dirs = np.radians(phi + turns)
        vec = np.stack([np.cos(dirs), np.sin(dirs)], axis=1)
        mids = P + 0.5 * s.growth_step_m * vec
        scores = np.array([self.scorer(m[0], m[1], float(axial(np.degrees(a))), row.spacing_index) for m, a in zip(mids, dirs)])
        finite = np.isfinite(scores)
        best = int(np.argmax(np.where(finite, scores, -np.inf))) if finite.any() else 0
        supported = bool(finite[best] and scores[best] >= s.growth_min_score)
        if supported:
            direction = vec[best]
        else:
            # Bridge: straight on along the row's longer-range direction.
            direction = row.outward_tangent(side, self.bridge_back_m)
        Q = P + s.growth_step_m * direction

        target = self._join_target(row, side, P, Q, direction)
        if target is not None:
            self._join(row, side, *target, supported)
            return
        segment = LineString([P, Q])
        if self._conflicts(row, segment, Q):
            ep.state = "conflict"
            return
        if not self.polygon.intersects(Point(Q)):
            hit = segment.intersection(self.boundary)
            pts = [np.array(g.coords[0]) for g in shapely.get_parts(hit) if not g.is_empty] if not hit.is_empty else []
            self._append(row, side, min(pts, key=lambda p: np.hypot(*(p - P))) if pts else Q, supported)
            ep.state = "boundary"
            return

        self._append(row, side, Q, supported)
        ep.steps += 1
        ep.grown_m += s.growth_step_m
        if supported:
            ep.gap_m, ep.tail = 0.0, 0
            ep.supported_steps += 1
        else:
            ep.gap_m += s.growth_step_m
            ep.tail += 1
            if ep.gap_m > s.max_gap_m + 1e-9:
                self._trim_tail(row, side, ep.tail)
                ep.tail = 0
                ep.state = "unsupported"
                return
        if ep.grown_m >= s.max_row_length_m:
            ep.state = "limit"

    @staticmethod
    def _append(row: Row, side: str, point: np.ndarray, supported: bool) -> None:
        if side == "end":
            row.points.append(np.asarray(point, dtype=float))
            row.seg_support.append(supported)
        else:
            row.points.insert(0, np.asarray(point, dtype=float))
            row.seg_support.insert(0, supported)

    @staticmethod
    def _trim_tail(row: Row, side: str, n: int) -> None:
        n = min(n, len(row.points) - 2)
        if n <= 0:
            return
        if side == "end":
            del row.points[-n:]
            del row.seg_support[-n:]
        else:
            del row.points[:n]
            del row.seg_support[:n]

    # -- joins and conflicts ------------------------------------------------------------------

    def _join_target(self, row: Row, side: str, P, Q, direction):
        if self._end_tree is None:
            return None
        reach = self.s.join_distance_frac * row.spacing_m
        lateral_max = self.s.link_lateral_frac * row.spacing_m
        normal = np.array([-direction[1], direction[0]])
        # Ends may have moved up to one step this round, so search a little wider.
        idx = self._end_tree.query_ball_point(Q, r=reach + 2 * self.s.growth_step_m)
        best = None
        for i in idx:
            other, other_side = self._ends[i]
            if other is row or not other.alive or other.endpoint(other_side).state not in FREE_STATES:
                continue
            E = other.tip(other_side)
            rel = E - P
            ahead = float(np.dot(rel, direction))
            lateral = abs(float(np.dot(rel, normal)))
            if lateral > lateral_max or ahead < -reach or ahead > self.s.growth_step_m + reach:
                continue
            dist = float(np.hypot(*rel))
            facing = -other.outward_tangent(other_side, self.back_m)
            if math.degrees(math.acos(float(np.clip(np.dot(facing, direction), -1, 1)))) > self.s.join_angle_deg:
                continue
            if best is None or dist < best[0]:
                best = (dist, other, other_side)
        return None if best is None else (best[1], best[2])

    def _join(self, row: Row, side: str, other: Row, other_side: str, supported: bool = True) -> None:
        if side == "start":
            row.reverse()  # joining end is now row.end
        if other_side == "end":
            other.reverse()  # joining end is now other.start
        joint = other.points[0]
        row.seg_support += [supported] + other.seg_support  # connector, then the other row
        row.points.extend(other.points)
        row.end = other.end
        row.seed_ids += other.seed_ids
        row.chain_ids += other.chain_ids
        row.joined_rows += [other.id] + other.joined_rows
        row.flags += [f for f in other.flags if f not in row.flags]
        other.alive = False
        self.joins.append(JoinRecord(row_id=row.id, merged_row_id=other.id, x=float(joint[0]), y=float(joint[1])))

    def _conflicts(self, row: Row, segment: LineString, Q) -> bool:
        overlap = self.s.link_lateral_frac * row.spacing_m
        for k in self._line_tree.query(segment.buffer(self.s.growth_step_m + overlap)):
            other = self._line_rows[int(k)]
            if other is row or not other.alive:
                continue
            line = other.line()
            if line.crosses(segment) or line.distance(Point(Q)) < overlap:
                return True
        return False
