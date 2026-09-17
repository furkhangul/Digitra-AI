"""V6 camera session with static, motion, and combined recognition paths."""
from time import monotonic, perf_counter

import numpy as np

from app.services.digitra_landmark.schemas import HandObservation
from app.services.digitra_landmark.unified_recognition import StableTextComposer
from app.services.digitra_tid import SmoothedTIDImageRecognizer, decode_image_data
from motion_recognition import DYNAMIC_LETTERS, MotionRecognizer, waiting
from hand_geometry import NPoseRecognizer


class V6RecognitionSession:
    def __init__(self, classifier):
        self.image = SmoothedTIDImageRecognizer(classifier.tid_image, minimum_confidence=.65,
                                               minimum_margin=.12, minimum_vote_ratio=.8)
        self.motion = MotionRecognizer(classifier.tid)
        self.n_pose = NPoseRecognizer()
        self.composer = StableTextComposer()
        # ``auto`` keeps the measured V6 image path and the temporal path
        # alive together.  This is the default for the camera test so a user
        # does not have to switch modes before making J/Ğ.
        self.recognition_mode = 'auto'
        self.last_result = waiting()
        self.hand_count = 0

    def reset_recognition(self):
        self.image.reset()
        self.motion.reset()
        self.n_pose.reset()
        self.last_result = waiting()
        self.hand_count = 0
        # Preserve text while clearing a pending or locked letter on mode change.
        text = self.composer.text
        self.composer.clear()
        self.composer.text = text

    @staticmethod
    def hands(raw):
        if not isinstance(raw, list) or len(raw) > 2 or any(not isinstance(item, dict) for item in raw):
            raise ValueError('En fazla iki el gönderilmelidir')
        observations = [HandObservation.from_dict(item) for item in raw]
        if any(not np.isfinite(hand.landmarks).all() for hand in observations):
            raise ValueError('El noktaları sonlu sayılar olmalıdır')
        return observations

    def update(self, payload):
        started, now = perf_counter(), monotonic()
        action = payload.get('action')
        added = None
        if action == 'set_recognition_mode':
            mode = payload.get('recognition_mode')
            if mode not in ('alphabet', 'motion', 'auto'):
                raise ValueError('Geçersiz tanıma modu')
            self.recognition_mode = mode
            self.reset_recognition()
        elif action:
            if action == 'space':
                self.composer.append_space()
            elif action == 'backspace':
                self.composer.backspace()
            elif action == 'clear':
                self.composer.clear()
                self.reset_recognition()
            elif action == 'append' and self.last_result['ready']:
                self.composer.append_prediction(self.last_result['label'])
            else:
                raise ValueError('Desteklenmeyen işlem')
        else:
            hands = self.hands(payload.get('hands', []))
            self.hand_count = len(hands)
            rows = payload.get('motion_frames', [])
            if not isinstance(rows, list) or len(rows) > 60:
                raise ValueError('Hareket paketi en fazla 60 kare içermelidir')
            aspect = float(payload.get('aspect_ratio', 1.0))
            motion_result = None
            if self.recognition_mode in ('motion', 'auto') and rows:
                # Parse the entire batch before mutating recognition state.
                frames = [(float(row['timestamp_ms']) / 1000, self.hands(row['hands'])) for row in rows]
                times = [frame[0] for frame in frames]
                if not np.isfinite(times).all() or any(b <= a for a, b in zip(times, times[1:])):
                    raise ValueError('Hareket kareleri zaman sırasıyla gönderilmelidir')
                for timestamp, observations in frames:
                    motion_result = self.motion.update(observations, timestamp, aspect)
                self.hand_count = len(frames[-1][1])
            elif self.recognition_mode == 'motion':
                self.motion.reset()
                motion_result = waiting(reason='COLLECTING_MOTION')

            n_result = self.n_pose.update(hands, now, aspect)
            if self.recognition_mode == 'motion':
                self.last_result = motion_result or waiting(reason='COLLECTING_MOTION')
            elif self.recognition_mode == 'auto':
                image_result = None
                if hands:
                    image = payload.get('image')
                    if isinstance(image, str) and image:
                        image_result = self.image.update(decode_image_data(image))
                        image_result.update(source='v6_image', confidence_kind='model')
                        if image_result['candidate'] in DYNAMIC_LETTERS:
                            image_result.update(ready=False, label='?', reason='USE_MOTION_MODE')
                        elif image_result['candidate'] == 'G' and (
                                self.motion.g_motion_pending or
                                (self.motion.samples and self.motion.samples[-1][0] - self.motion.samples[0][0] < 1.3)):
                            image_result.update(ready=False, label='?', reason='CHECKING_G_MOTION')

                # A completed trajectory always wins.  While a two-hand
                # trajectory is still being collected, suppress a static
                # dynamic-letter guess but continue to allow ordinary static
                # letters through the V6 image model.
                if motion_result and motion_result.get('ready'):
                    self.last_result = motion_result
                elif n_result is not None:
                    self.last_result = n_result
                elif motion_result and len(hands) >= 2 and image_result and image_result.get('candidate') in DYNAMIC_LETTERS:
                    self.last_result = motion_result
                elif image_result is not None:
                    self.last_result = image_result
                elif not hands:
                    self.image.reset()
                    self.motion.reset()
                    self.last_result = waiting(reason='NO_HAND')
                elif motion_result is not None:
                    self.last_result = motion_result
                else:
                    self.last_result = waiting(reason='COLLECTING_MOTION')
            elif hands:
                image = payload.get('image')
                if not isinstance(image, str) or not image:
                    raise ValueError('V6 testi için el görüntüsü gerekli')
                self.last_result = self.image.update(decode_image_data(image))
                self.last_result.update(source='v6_image', confidence_kind='model')
                if self.last_result['candidate'] in DYNAMIC_LETTERS:
                    self.last_result.update(ready=False, label='?', reason='USE_MOTION_MODE')
                if n_result is not None:
                    self.last_result = n_result
            else:
                self.image.reset()
                self.motion.reset()
                self.last_result = waiting(reason='NO_HAND')
            # A user may hold G before beginning Ğ's movement. If G was
            # committed during that same unreleased gesture, finish it as Ğ
            # rather than leaving the text locked on the static base shape.
            if (self.last_result['ready'] and self.last_result['label'] == 'Ğ' and
                    self.composer.locked_label == 'G' and self.composer.text.endswith('G') and
                    self.composer.no_hands_since is None):
                self.composer.text = self.composer.text[:-1] + 'Ğ'
                self.composer.locked_label = 'Ğ'
                added = 'Ğ'
            else:
                added = self.composer.update(self.last_result['label'], self.last_result['ready'],
                                             self.hand_count > 0, now)
        result = self.last_result
        ready = bool(result['ready'])
        return dict(model_ready=True, model='digitra-tid-augmented', version='6.0.0',
                    scope='tid_motion_experimental' if self.recognition_mode == 'motion' else 'tid_alphabet_image',
                    mode='tid_motion' if result.get('confidence_kind') == 'trajectory_rule' or (self.recognition_mode == 'motion' and self.hand_count) else ('tid_image' if self.hand_count else 'waiting'),
                    recognition_mode=self.recognition_mode, motion_supported=True,
                    hands_detected=self.hand_count, prediction=result['label'] if ready else None,
                    candidate=result['candidate'], confidence=result['confidence'], margin=result['margin'],
                    ready=ready, frames=result['frames'], candidates=result['candidates'],
                    confidence_kind=result.get('confidence_kind', 'model'), source=result.get('source', 'v6_image'),
                    motion_status=result.get('reason'), text=self.composer.text, added=added,
                    locked=bool(self.composer.locked_label), progress=self.composer.progress(now),
                    reject_reason=None if ready else ('NO_HAND' if not self.hand_count else result.get('reason', 'LOW_CONFIDENCE')),
                    latency_ms=round((perf_counter() - started) * 1000, 2))
