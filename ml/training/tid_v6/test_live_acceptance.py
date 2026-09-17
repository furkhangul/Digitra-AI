"""Camera acceptance must not turn repeated uncertain guesses into text."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'apps/api'))
from app.services.digitra_tid.recognizer import SmoothedTIDImageRecognizer


class FixedClassifier:
    labels=['A','B','C']
    recommended_confidence=0.0

    def __init__(self,probabilities):self.probabilities=np.array(probabilities)
    def probability_vector(self,image):return self.probabilities


def recognize(probabilities):
    model=SmoothedTIDImageRecognizer(FixedClassifier(probabilities),minimum_confidence=.65,minimum_margin=.12,minimum_vote_ratio=.8)
    for _ in range(10):result=model.update(None)
    return result


def test_repeated_low_confidence_is_visible_but_not_ready():
    result=recognize([.37,.36,.27])
    assert result['candidate']=='A' and result['confidence']==.37
    assert result['ready'] is False and result['label']=='?'


def test_clear_stable_prediction_is_ready():
    result=recognize([.9,.06,.04])
    assert result['ready'] is True and result['label']=='A'


def test_margin_rejects_a_close_second_candidate_even_when_confidence_passes():
    model=SmoothedTIDImageRecognizer(FixedClassifier([.54,.45,.01]),minimum_confidence=.5,minimum_margin=.12)
    for _ in range(5):result=model.update(None)
    assert result['ready'] is False


def test_default_policy_keeps_existing_runtime_behavior():
    model=SmoothedTIDImageRecognizer(FixedClassifier([.37,.36,.27]))
    for _ in range(5):result=model.update(None)
    assert result['ready'] is True
