"""Experimental TİD motion testing, separate from V6's static image score.

The six-class personal temporal checkpoint is reused without retraining. J is
an ordered geometric path check and Ğ has a local index-bob gate, both matching
the site's authored two-hand signs. They are not the legacy ASL rule and are
not calibrated webcam accuracy claims. Nothing is recorded.
"""
from collections import deque

import numpy as np

from app.services.digitra_landmark.features import extract_features
from app.services.digitra_landmark.hand_recovery import (
    choose_orientation_probabilities, mirror_hands,
)
from hand_geometry import image_points, palm_size, matches_g_pose

DYNAMIC_LETTERS = frozenset('ÇĞİJÖŞÜ')


def waiting(frames=0, reason='COLLECTING_MOTION'):
    return dict(label='?', candidate='?', confidence=0.0, margin=0.0,
                ready=False, frames=frames, candidates=[], reason=reason,
                confidence_kind='model', source='personal_temporal')


def j_coordinates(anchor, pointer):
    """Index tip -> inside L corner -> thumb tip, in the anchor hand's frame."""
    size = palm_size(anchor)
    origin = anchor[5]
    vertical = anchor[8] - origin
    length = float(np.linalg.norm(vertical))
    if length < .8 * size:
        return None
    vertical /= length
    side = anchor[4] - origin
    side -= np.dot(side, vertical) * vertical
    width = float(np.linalg.norm(side))
    # An open L is needed; a fist, C, or two extended fingers is not a J anchor.
    if width < .45 * size or any(np.linalg.norm(anchor[i] - anchor[0]) > 1.55 * size
                               for i in (12, 16, 20)):
        return None
    side /= width
    point = pointer[8] - origin
    thumb_y = float(np.dot(anchor[4] - origin, vertical) / length)
    return np.array([np.dot(point, side) / width,
                     np.dot(point, vertical) / length, thumb_y])


def matches_j_path(samples):
    """Require ordered coverage, a bend and a completed stroke; no static J."""
    if len(samples) < 7 or samples[-1][0] - samples[0][0] < .45:
        return False
    for anchor_slot in (0, 1):
        path = [j_coordinates(points[anchor_slot], points[1 - anchor_slot])
                for _, points in samples]
        for start in range(len(path) - 6):
            first = path[start]
            if first is None or np.linalg.norm(first[:2] - [0, 1]) > .38:
                continue
            segment = path[start:]
            duration = samples[-1][0] - samples[start][0]
            if not .45 <= duration <= 3.4 or any(p is None for p in segment):
                continue
            values = np.stack(segment)
            x, y, thumb_y = values.T
            if np.linalg.norm(values[-1, :2] - [1, thumb_y[-1]]) > .38:
                continue
            # First descend alongside the index, then move out along the thumb.
            corners = np.flatnonzero((y < .42) & (x < .60) & (x > -.35))
            if not len(corners) or corners[0] == len(values) - 1:
                continue
            corner = int(corners[0])
            if np.max(x[:corner + 1]) > .60 or np.min(y[:corner + 1]) < -.40:
                continue
            if np.any(x < -.45) or np.any(x > 1.45) or np.any(y > 1.4) or np.any(y < -.55):
                continue
            # Reject reversals and side-to-side waving within the stroke.
            if np.sum(np.maximum(np.diff(y[:corner + 1]), 0)) > .35:
                continue
            if np.sum(np.maximum(-np.diff(x[corner:]), 0)) > .35:
                continue
            return True
    return False


def articulation(points):
    """Finger motion in each palm frame excludes camera pan/zoom and arm sway."""
    output = []
    for hand in points:
        origin = hand[0]
        axis = hand[9] - origin
        axis /= max(float(np.linalg.norm(axis)), 1e-6)
        side = np.array([-axis[1], axis[0]])
        local = (hand[[4, 8, 12, 16, 20]] - origin) @ np.stack([axis, side], axis=1)
        output.append(local / palm_size(hand))
    return np.stack(output)


def matches_g_index_bob(movement):
    """Detect Ğ's small repeated index bob without relying on the old model.

    The authored sign moves one index while the other four fingertips and the
    two palms stay still.  Measuring in each palm's local frame makes this
    robust to camera translation, distance, and a hand-order swap.  A repeated
    direction change is required so a single snap or an accidental reach is
    not promoted to Ğ.
    """
    values = np.asarray(movement, dtype=float)
    if values.ndim != 4 or values.shape[1:] != (2, 5, 2):
        return False
    for hand in range(2):
        index = values[:, hand, 1, :]
        # Project on the actual motion axis instead of a camera-aligned axis.
        # A user can bend farther than the five-degree teaching animation.
        _, _, axes = np.linalg.svd(index - np.median(index, axis=0), full_matrices=False)
        signal = index @ axes[0]
        amplitude = float(np.quantile(signal, .95) - np.quantile(signal, .05))
        residual = np.ptp(index @ axes[1])
        others = values[:, hand, [0, 2, 3, 4], :]
        other_motion = np.linalg.norm(np.quantile(others, .95, axis=0) -
                                      np.quantile(others, .05, axis=0), axis=-1).max()
        if (amplitude < .055 or amplitude > .65 or residual > amplitude * .65
                or amplitude < 2.5 * max(float(other_motion), .008)):
            continue
        # Hysteresis counts meaningful reversals, not individual noisy frame
        # derivatives. Two reversals require an out/back/out movement.
        threshold = max(.025, amplitude * .28)
        low = high = signal[0]
        direction = turns = 0
        for value in signal[1:]:
            if direction == 0:
                low, high = min(low, value), max(high, value)
                if value - low >= threshold:
                    direction, high = 1, value
                elif high - value >= threshold:
                    direction, low = -1, value
            elif direction > 0:
                high = max(high, value)
                if high - value >= threshold:
                    direction, low, turns = -1, value, turns + 1
            else:
                low = min(low, value)
                if value - low >= threshold:
                    direction, high, turns = 1, value, turns + 1
        if turns >= 2:
            return True
    return False


class MotionRecognizer:
    WINDOW_SECONDS = 2.6
    MAX_GAP_SECONDS = .4

    def __init__(self, classifier):
        self.classifier = classifier
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=150)
        self.features = deque(maxlen=150)
        self.articulations = deque(maxlen=150)
        self.g_shapes = deque(maxlen=150)
        self.votes = deque(maxlen=3)
        self.previous = None
        self.last_time = None
        self.last_evaluation = -float('inf')
        self.latched = None
        self.latch_until = 0.0
        self.g_motion_pending = False

    def update(self, hands, now, aspect=1.0):
        if not np.isfinite(now) or not np.isfinite(aspect) or not .2 <= aspect <= 5:
            self.reset()
            return waiting(reason='INVALID_FRAME')
        if self.last_time is not None and (now <= self.last_time or now - self.last_time > self.MAX_GAP_SECONDS):
            self.reset()
        self.last_time = now
        if len(hands) != 2:
            self.g_motion_pending = False
            # Do not invent motion by copying an occluded hand into new frames.
            if not hands:
                self.reset()
            elif self.samples and now - self.samples[-1][0] > self.MAX_GAP_SECONDS:
                self.reset()
            return waiting(reason='TWO_HANDS_REQUIRED')

        points = np.stack([image_points(hand, aspect) for hand in hands])
        if not np.isfinite(points).all():
            self.reset()
            return waiting(reason='INVALID_FRAME')
        # MediaPipe may swap output order/handedness during contact.
        if self.previous is not None:
            cost = [np.linalg.norm(points[order, 0] - self.previous[:, 0], axis=1).sum()
                    for order in ([0, 1], [1, 0])]
            if cost[1] < cost[0]:
                points = points[::-1].copy()
                hands = hands[::-1]
            size = max(palm_size(points[0]), palm_size(points[1]))
            if min(cost) > 4 * size:
                self.reset()
                self.last_time = now
        self.previous = points
        self.samples.append((now, points))
        self.articulations.append(articulation(points))
        self.g_shapes.append(matches_g_pose(points))
        self.features.append((now, extract_features(hands).vector,
                              extract_features(mirror_hands(hands)).vector))
        while self.samples and now - self.samples[0][0] > 3.4:
            self.samples.popleft()
            self.features.popleft()
            self.articulations.popleft()
            self.g_shapes.popleft()

        if self.latched is not None and now <= self.latch_until:
            return self.latched
        self.latched = None
        if matches_j_path(list(self.samples)):
            result = dict(label='J', candidate='J', confidence=0.0, margin=0.0,
                          ready=True, frames=len(self.samples), candidates=[], reason=None,
                          confidence_kind='trajectory_rule', source='tid_j_path_rule')
            return self._latch(result, now)

        duration = now - self.samples[0][0]
        result = waiting(len(self.samples))
        if duration < .8 or len(self.samples) < 8:
            return result
        # A few noisy landmarks or moving both arms together are insufficient.
        # Cache palm-local features once per frame instead of recomputing the
        # whole 2.6-second history for every incoming camera frame.
        recent = [(movement, shape) for (time, _), movement, shape in
                  zip(self.samples, self.articulations, self.g_shapes)
                  if now - time <= self.WINDOW_SECONDS]
        movement = np.stack([item[0] for item in recent])
        # A three-frame median suppresses detector jitter while retaining the
        # small, sustained index bob in Ğ. Raw features still retain snap timing.
        movement = np.stack([np.median(movement[max(0, i-1):i+2], axis=0)
                             for i in range(len(movement))])
        excursion = np.linalg.norm(np.quantile(movement, .9, axis=0) -
                                   np.quantile(movement, .1, axis=0), axis=-1)
        g_shape = np.mean([item[1] for item in recent]) >= .75
        index_excursion = float(excursion[:, 1].max())
        self.g_motion_pending = bool(g_shape and index_excursion >= .04)
        if g_shape and matches_g_index_bob(movement):
            result = dict(label='Ğ', candidate='Ğ', confidence=0.0, margin=0.0,
                          ready=True, frames=len(self.samples), candidates=[], reason=None,
                          confidence_kind='trajectory_rule', source='tid_g_index_bob_rule')
            return self._latch(result, now)
        if duration < self.WINDOW_SECONDS or len(self.samples) < 18:
            return result
        # The authored Ğ's five-degree index bob spans about .08 palm units.
        if float(excursion.max()) < .055:
            self.votes.clear()
            result['reason'] = 'MOTION_REQUIRED'
            return result
        # Score at most 10 times/s even when the browser supplies 30 FPS batches.
        if now - self.last_evaluation < .10:
            return result
        self.last_evaluation = now
        times = np.array([row[0] for row in self.features])
        grid = np.linspace(now - self.WINDOW_SECONDS, now, self.classifier.sequence_frames)
        sequences = []
        for orientation in (1, 2):
            features = np.stack([row[orientation] for row in self.features])
            sequences.append(np.column_stack([np.interp(grid, times, features[:, i])
                                               for i in range(features.shape[1])]))
        probabilities, _ = choose_orientation_probabilities(
            self.classifier.probabilities(sequences[0]),
            self.classifier.probabilities(sequences[1]),
        )
        order = np.argsort(probabilities)[::-1]
        confidence = float(probabilities[order[0]])
        margin = confidence - float(probabilities[order[1]])
        candidate = str(self.classifier.classes[order[0]])
        self.votes.append(candidate if confidence >= .70 and margin >= .18 else '?')
        ready = len(self.votes) == 3 and all(item == candidate for item in self.votes)
        result.update(candidate=candidate, confidence=confidence, margin=margin,
                      candidates=[dict(label=str(self.classifier.classes[i]), confidence=float(probabilities[i]))
                                  for i in order[:3]],
                      ready=ready, label=candidate if ready else '?',
                      reason=None if ready else 'UNCERTAIN_MOTION')
        return self._latch(result, now) if ready else result

    def _latch(self, result, now):
        # Keep a completed gesture available long enough for the existing text
        # dwell gate. Losing either hand hides it; a release clears all history.
        self.latched = result
        self.latch_until = now + 1.6
        self.votes.clear()
        return result
