"""Conservative checks for the site's two-hand N and G shapes.

These are geometric checks, not trained probabilities or webcam accuracy scores.
Coordinates use the unmirrored camera's aspect-corrected image plane.
"""
import numpy as np


def image_points(hand, aspect):
    return hand.landmarks[:, :2].astype(float) * np.array([aspect, 1.0])


def palm_size(points):
    return max(float(np.median([np.linalg.norm(points[a] - points[b])
                               for a, b in [(0, 9), (5, 17), (0, 5), (0, 17)]])), 1e-6)


def finger_extension(hand):
    # Open fingertips extend beyond their PIP joint; folded fingers return
    # towards the wrist. Palm normalization removes camera-distance changes.
    return np.array([(np.linalg.norm(hand[b + 3] - hand[0]) -
                      np.linalg.norm(hand[b + 1] - hand[0])) / palm_size(hand)
                     for b in (5, 9, 13, 17)])


def matches_n_pose(points):
    if np.shape(points) != (2, 21, 2) or not np.isfinite(points).all():
        return False
    extensions = np.stack([finger_extension(hand) for hand in points])
    # Exactly two extended fingers on one hand, one on the other. This is
    # deliberately order-independent: MediaPipe swaps hands during contact.
    if not np.all(extensions[:, 0] > .18) or not np.all(extensions[:, 2:] < -.06):
        return False
    if not (np.max(extensions[:, 1]) > .18 and np.min(extensions[:, 1]) < -.08):
        return False
    directions = points[:, 8] - points[:, 5]
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-6)
    # N's three legs point down together. A's crossing bar and Y's upright
    # fork must not pass just because their extended-finger counts match.
    if np.any(directions[:, 1] < .40) or np.dot(*directions) < .45:
        return False
    size = max(palm_size(hand) for hand in points)
    return bool(np.linalg.norm(points[0, 8] - points[1, 8]) < .65 * size and
                np.linalg.norm(points[0, 0] - points[1, 0]) < 2.6 * size)


def matches_g_pose(points):
    if np.shape(points) != (2, 21, 2) or not np.isfinite(points).all():
        return False
    sizes = [palm_size(hand) for hand in points]
    extensions = np.stack([finger_extension(hand) for hand in points])
    # Two open thumb/index curves, three folded fingers, nearby thumbs.
    # Prevent an unrelated pointing/waving gesture from becoming Ğ.
    return bool(np.all(extensions[:, 0] > .04) and
                np.all(extensions[:, 1:] < -.06) and
                all(.40 < np.linalg.norm(h[8] - h[4]) / s < 1.65 for h, s in zip(points, sizes)) and
                np.linalg.norm(points[0, 4] - points[1, 4]) < .65 * max(sizes))


class NPoseRecognizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.since = None
        self.last_time = None
        self.frames = 0

    def update(self, hands, now, aspect):
        if (len(hands) != 2 or not np.isfinite(aspect) or not .2 <= aspect <= 5 or
                not matches_n_pose(np.stack([image_points(h, aspect) for h in hands]))):
            self.reset()
            return None
        if self.last_time is None or now <= self.last_time or now - self.last_time > .4:
            self.since = now
            self.frames = 0
        self.last_time = now
        self.frames += 1
        ready = self.frames >= 3 and now - self.since >= .45
        return dict(label='N' if ready else '?', candidate='N', confidence=0.0,
                    margin=0.0, ready=ready, frames=self.frames, candidates=[],
                    reason=None if ready else 'HOLD_N_POSE',
                    confidence_kind='pose_rule', source='tid_n_pose_rule')
