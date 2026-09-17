"""Behavior tests with generated landmarks; these are NOT webcam accuracy tests."""
import sys
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'apps/api'))
from app.services.digitra_landmark.schemas import HandObservation
from live_session import V6RecognitionSession
from motion_recognition import MotionRecognizer, articulation, matches_g_index_bob, matches_j_path


def pair(pointer=(0, 1.3)):
    # An upright L, folded middle/ring/pinky; a separate moving index pointer.
    anchor = np.array([
        [.25, -1], [.5, -.65], [.72, -.25], [.95, -.05], [1.2, .1],
        [0, 0], [0, .5], [0, .95], [0, 1.3],
        [.25, 0], [.3, .3], [.3, -.15], [.3, -.4],
        [.5, -.1], [.55, .15], [.55, -.2], [.55, -.45],
        [.7, -.2], [.75, .05], [.75, -.25], [.75, -.5],
    ])
    tracer = anchor.copy() * .75 + [2, -.4]
    tracer[8] = pointer
    return np.stack([anchor, tracer])


def j_frames(count=31, reverse=False):
    # Down the index, then round towards the extended thumb. Independent path.
    route = np.array([[0, 1.3], [.05, .6], [.12, .15], [.6, .05], [1.2, .1]])
    t = np.linspace(0, 4, count)
    points = np.column_stack([np.interp(t, np.arange(5), route[:, d]) for d in (0, 1)])
    if reverse:
        points = points[::-1]
    return [(i * 2.0 / (count - 1), pair(p)) for i, p in enumerate(points)]


def observations(points, aspect=1.0):
    result = []
    for i, hand in enumerate(points):
        xyz = np.column_stack([hand * .10 / [aspect, 1] + [.30, .4], np.zeros(21)])
        result.append(HandObservation(xyz, handedness=['Right', 'Left'][i], handedness_score=.99))
    return result


class TemporalStub:
    classes = np.array(list('ÇĞİÖŞÜ'))
    sequence_frames = 30

    def __init__(self):
        self.sequences = []

    def probabilities(self, sequence):
        self.sequences.append(sequence.copy())
        assert sequence.shape == (30, 264)
        return np.array([.02, .90, .02, .02, .02, .02])


def test_j_requires_order_bend_and_complete_path():
    assert matches_j_path(j_frames())
    assert not matches_j_path(j_frames(reverse=True))
    assert not matches_j_path(j_frames()[:14])
    assert not matches_j_path([(i / 15, pair()) for i in range(31)])
    diagonal = [(i / 15, pair((1.2 * i / 30, 1.3 - 1.2 * i / 30))) for i in range(31)]
    assert not matches_j_path(diagonal)


@pytest.mark.parametrize('scale,angle,mirror,aspect', [(1, 0, 1, 1), (.4, .3, -1, 16/9), (1.3, -.5, 1, 9/16)])
def test_j_survives_distance_mirror_rotation_and_hand_order_changes(scale, angle, mirror, aspect):
    model = MotionRecognizer(TemporalStub())
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    accepted = []
    for i, (time, points) in enumerate(j_frames()):
        transformed = (points * [mirror, 1]) @ rotation * scale + [.2, -.1]
        hands = observations(transformed, aspect)
        if i % 2:
            hands.reverse()
        result = model.update(hands, time, aspect)
        if result['ready']:
            accepted.append(result)
    assert accepted and accepted[0]['label'] == 'J'
    assert accepted[0]['confidence_kind'] == 'trajectory_rule'
    assert accepted[0]['confidence'] == 0  # Never fabricate a percentage for a rule.


def test_static_jitter_and_arm_sway_do_not_become_soft_g_even_with_confident_classifier():
    classifier = TemporalStub()
    model = MotionRecognizer(classifier)
    rng = np.random.default_rng(18)
    for i in range(130):
        points = pair() * (1 + .1 * np.sin(i / 30)) + [np.sin(i / 20) * .2, .1]
        points += rng.normal(0, .002, points.shape)
        assert not model.update(observations(points), i / 30)['ready']
    assert classifier.sequences == []


@pytest.mark.parametrize('fps', [10, 30])
def test_temporal_model_gets_resampled_motion_and_can_accept_soft_g(fps):
    classifier = TemporalStub()
    model = MotionRecognizer(classifier)
    accepted = []
    for i in range(4 * fps):
        points = pair((2, 1))
        points[1, 8, 1] += .20 * np.sin(i / fps * 8)
        result = model.update(observations(points), i / fps)
        if result['ready']:
            accepted.append(result['label'])
    assert classifier.sequences and 'Ğ' in accepted


def test_authored_soft_g_can_be_completed_by_local_index_bob_rule():
    sequences = json.loads((Path(__file__).resolve().parents[3] / 'artifacts/tid-v6/motion/authored_sequences.json').read_text(encoding='utf-8'))['sequences']
    movement = np.stack([articulation(np.array(row['points'])) for row in sequences['Ğ']])
    assert matches_g_index_bob(movement)


def test_hand_loss_and_long_pause_discard_partial_j():
    model = MotionRecognizer(TemporalStub())
    frames = j_frames()
    for time, points in frames[:15]:
        model.update(observations(points), time)
    assert not model.update([], 1.01)['ready']
    for time, points in frames[15:]:
        assert not model.update(observations(points), time + 2)['ready']
    assert not model.update(observations(frames[-1][1]), 20)['ready']


def test_missing_hand_hides_completed_event_immediately():
    model = MotionRecognizer(TemporalStub())
    for time, points in j_frames():
        result = model.update(observations(points), time)
    assert result['ready']
    assert not model.update(observations(pair())[:1], 2.1)['ready']


class ImageStub:
    labels = ['J', 'A', 'G']
    recommended_confidence = 0

    def __init__(self):
        self.calls = 0

    def probability_vector(self, image):
        self.calls += 1
        return np.array([.96, .02, .02])


def payload_hands(points):
    return [dict(landmarks=h.landmarks.tolist(), handedness=h.handedness,
                 handedness_score=.99) for h in observations(points)]


def test_motion_session_consumes_full_rate_landmarks_instead_of_image_branch(monkeypatch):
    image = ImageStub()
    session = V6RecognitionSession(SimpleNamespace(tid_image=image, tid=TemporalStub()))
    session.composer.text = 'A'
    session.update(dict(action='set_recognition_mode', recognition_mode='motion'))
    clock = [0.0]
    monkeypatch.setattr('live_session.monotonic', lambda: clock[0])
    frames = j_frames()
    outputs = []
    for offset in range(0, len(frames), 3):
        batch = frames[offset:offset+3]
        clock[0] = batch[-1][0]
        outputs.append(session.update(dict(hands=payload_hands(batch[-1][1]), image='unused crop',
            motion_frames=[dict(timestamp_ms=t*1000, hands=payload_hands(p)) for t, p in batch])))
    assert any(row['prediction'] == 'J' for row in outputs)
    # Hold the end pose after a completed gesture so the dwell gate can append.
    for i in range(1, 17):
        clock[0] = 2 + i * .1
        result = session.update(dict(hands=payload_hands(frames[-1][1]),
            motion_frames=[dict(timestamp_ms=clock[0]*1000, hands=payload_hands(frames[-1][1]))]))
    assert result['text'] == 'AJ' and image.calls == 0
    session.update(dict(action='set_recognition_mode', recognition_mode='alphabet'))
    assert session.composer.text == 'AJ' and not session.last_result['ready']


def test_auto_mode_keeps_motion_path_alive_without_an_image():
    image = ImageStub()
    session = V6RecognitionSession(SimpleNamespace(tid_image=image, tid=TemporalStub()))
    frames = j_frames()
    outputs = []
    for offset in range(0, len(frames), 3):
        batch = frames[offset:offset + 3]
        outputs.append(session.update(dict(
            hands=payload_hands(batch[-1][1]),
            motion_frames=[dict(timestamp_ms=t * 1000, hands=payload_hands(p)) for t, p in batch],
        )))
    assert any(row['prediction'] == 'J' for row in outputs)
    assert image.calls == 0


def test_static_image_cannot_complete_a_dynamic_letter(monkeypatch):
    monkeypatch.setattr('live_session.decode_image_data', lambda _: None)
    session = V6RecognitionSession(SimpleNamespace(tid_image=ImageStub(), tid=TemporalStub()))
    for _ in range(8):
        result = session.update(dict(hands=payload_hands(pair()), image='fake'))
    assert result['candidate'] == 'J' and not result['ready']
    assert result['motion_status'] == 'USE_MOTION_MODE'


def test_invalid_frame_batch_is_rejected_before_updating_motion():
    session = V6RecognitionSession(SimpleNamespace(tid_image=ImageStub(), tid=TemporalStub()))
    session.update(dict(action='set_recognition_mode', recognition_mode='motion'))
    with pytest.raises(ValueError):
        session.update(dict(hands=[], motion_frames=[dict(timestamp_ms=t, hands=[]) for t in (2, 1)]))
    assert not session.motion.samples


def test_authored_teaching_animations_match_j_only_and_keep_small_soft_g():
    path = Path(__file__).resolve().parents[3] / 'artifacts/tid-v6/motion/authored_sequences.json'
    sequences = json.loads(path.read_text(encoding='utf-8'))['sequences']
    for letter, rows in sequences.items():
        samples = [(row['time'], np.array(row['points'])) for row in rows]
        matched = any(matches_j_path(samples[:end]) for end in range(7, len(samples)))
        assert matched == (letter == 'J'), letter
    classifier = TemporalStub()
    model = MotionRecognizer(classifier)
    outputs = [model.update(observations(np.array(row['points'])), row['time']) for row in sequences['Ğ']]
    assert any(result['label'] == 'Ğ' for result in outputs)
