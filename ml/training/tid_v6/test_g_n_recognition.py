"""Regression checks; generated geometry is NOT a human accuracy benchmark."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_motion_recognition import observations, payload_hands, TemporalStub
from hand_geometry import matches_n_pose, NPoseRecognizer
from motion_recognition import MotionRecognizer
from live_session import V6RecognitionSession

SEQUENCES = json.loads((Path(__file__).resolve().parents[3] /
    'artifacts/tid-v6/motion/authored_sequences.json').read_text(encoding='utf-8'))['sequences']


class UncertainTemporal(TemporalStub):
    def probabilities(self, sequence):
        return np.full(6, 1 / 6)


def transformed(points, scale=1, angle=0, mirror=1):
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (points * [mirror, 1]) @ rotation * scale + [.2, -.1]


@pytest.mark.parametrize('scale,angle,mirror,aspect', [(1, 0, 1, 1), (.35, .3, -1, 16/9), (1.4, -.35, 1, 9/16)])
def test_n_geometry_separates_all_other_authored_letters(scale, angle, mirror, aspect):
    for letter, rows in SEQUENCES.items():
        for row in rows:
            points = transformed(np.array(row['points']), scale, angle, mirror)
            assert matches_n_pose(points) == (letter == 'N'), letter
    detector = NPoseRecognizer()
    points = transformed(np.array(SEQUENCES['N'][0]['points']), scale, angle, mirror)
    for i in range(10):
        hands = observations(points, aspect)
        if i % 2:
            hands.reverse()
        result = detector.update(hands, i / 10, aspect)
    assert result['ready'] and result['label'] == 'N'
    assert result['confidence'] == 0 and result['confidence_kind'] == 'pose_rule'
    assert detector.update(hands[:1], 1.0, aspect) is None
    assert not detector.update(hands, 1.1, aspect)['ready']


def test_n_requires_nearby_tips_and_consistent_frames():
    points = np.array(SEQUENCES['N'][0]['points'])
    separated = points.copy()
    separated[1] += [3, 0]
    assert not matches_n_pose(separated)
    detector = NPoseRecognizer()
    assert not detector.update(observations(points), 0, 1)['ready']
    assert not detector.update(observations(points), 1, 1)['ready']
    assert detector.update([], 1.1, 1) is None


def soft_g_frames(amplitude=1, speed=1, fps=30, angle=0, mirror=1, noise=0):
    rows = SEQUENCES['Ğ']
    base = np.array(rows[0]['points'])
    times = np.array([row['time'] for row in rows])
    values = np.array([row['points'] for row in rows])
    rng = np.random.default_rng(37)
    for i in range(int(times[-1] / speed * fps) + 1):
        time = i / fps
        index = min(int(time * speed * 30), len(rows) - 1)
        points = base + (values[index] - base) * amplitude
        points += rng.normal(0, noise, points.shape)
        yield time, transformed(points, .5, angle, mirror)


@pytest.mark.parametrize('amplitude,speed,fps,angle,mirror,noise', [
    (1, 1, 30, 0, 1, 0), (3, 1, 15, .3, -1, .002),
    (2, .55, 10, -.35, 1, .002), (.8, 1.3, 30, .2, -1, .001),
])
def test_soft_g_accepts_different_motion_sizes_speeds_and_mirrors(amplitude, speed, fps, angle, mirror, noise):
    detector = MotionRecognizer(UncertainTemporal())
    accepted = []
    for i, (time, points) in enumerate(soft_g_frames(amplitude, speed, fps, angle, mirror, noise)):
        hands = observations(points, 16/9)
        if i % 2:
            hands.reverse()
        result = detector.update(hands, time, 16/9)
        if result['ready']:
            accepted.append(result)
    assert accepted
    assert all(row['label'] == 'Ğ' and row['source'] == 'tid_g_index_bob_rule' for row in accepted)


@pytest.mark.parametrize('kind', ['static', 'jitter', 'whole_hand_sway', 'single_bend', 'wrong_shape'])
def test_soft_g_rejects_non_g_movement(kind):
    detector = MotionRecognizer(UncertainTemporal())
    rng = np.random.default_rng(17)
    for i in range(110):
        points = np.array(SEQUENCES['N' if kind == 'wrong_shape' else 'G'][0]['points'])
        if kind == 'jitter':
            points += rng.normal(0, .003, points.shape)
        elif kind == 'whole_hand_sway':
            points = transformed(points, 1 + .1 * np.sin(i / 15), angle=.1 * np.sin(i / 20))
        elif kind == 'single_bend':
            points[1, 8, 1] += .15 * min(i / 80, 1)
        elif kind == 'wrong_shape':
            points[1, 8, 1] += .10 * np.sin(i / 7)
        result = detector.update(observations(points), i / 30)
        assert not result['ready'], (kind, result)


class ConfusedImage:
    labels = ['M', 'N', 'G']
    recommended_confidence = 0

    def __init__(self, letter):
        self.letter = letter

    def probability_vector(self, image):
        return np.array([.9 if ch == self.letter else .05 for ch in self.labels])


def session_for(letter, monkeypatch):
    monkeypatch.setattr('live_session.decode_image_data', lambda _: None)
    return V6RecognitionSession(SimpleNamespace(tid_image=ConfusedImage(letter), tid=UncertainTemporal()))


def send_frame(session, points, time, clock):
    clock[0] = time
    return session.update(dict(hands=payload_hands(points), image='test',
        motion_frames=[dict(timestamp_ms=time * 1000, hands=payload_hands(points))]))


def test_n_overrides_confused_image_but_m_does_not_pass(monkeypatch):
    clock = [0]
    monkeypatch.setattr('live_session.monotonic', lambda: clock[0])
    for letter in ('N', 'M'):
        session = session_for('M', monkeypatch)
        points = np.array(SEQUENCES[letter][0]['points'])
        for i in range(26):
            result = send_frame(session, points, i / 10, clock)
        assert result['prediction'] == letter
        assert result['text'] == letter


def test_completed_soft_g_replaces_its_held_g_base_in_translation(monkeypatch):
    clock = [0]
    monkeypatch.setattr('live_session.monotonic', lambda: clock[0])
    session = session_for('G', monkeypatch)
    base = np.array(SEQUENCES['G'][0]['points'])
    for i in range(32):
        result = send_frame(session, base, i / 10, clock)
    assert result['text'] == 'G'
    # Same camera position, no release: begin motion after holding its base.
    for time, points in soft_g_frames(amplitude=2):
        # Undo the fixture helper's global transform to avoid a camera jump.
        points = (points - [.2, -.1]) / .5
        result = send_frame(session, points, 3.2 + time, clock)
    assert result['text'] == 'Ğ'


def test_soft_g_never_writes_g_when_motion_begins_immediately(monkeypatch):
    clock = [0]
    monkeypatch.setattr('live_session.monotonic', lambda: clock[0])
    session = session_for('G', monkeypatch)
    outputs = [send_frame(session, points, time, clock) for time, points in soft_g_frames(amplitude=2)]
    assert all('G' not in row['text'] for row in outputs)
    assert any(row['prediction'] == 'Ğ' for row in outputs)
