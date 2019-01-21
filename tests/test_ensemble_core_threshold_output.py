import json

from rasa_core import utils
from rasa_core.domain import Domain
from rasa_core.trackers import DialogueStateTracker
from rasa_core.policies import Policy
from rasa_core.policies.ensemble import SimplePolicyEnsemble
from rasa_core.policies.fallback import FallbackPolicy

CONF_PRED = 1
DOMAIN_PATH = "data/test_domains/default_with_fallback.yml"


class WorkingPolicy(Policy):
    @classmethod
    def load(cls, path):
        return WorkingPolicy()

    def persist(self, path):
        pass

    def train(self, training_trackers, domain, **kwargs):
        pass

    def predict_action_probabilities(self, tracker, domain):
        """Predicts always a predefined confidence vector"""
        global CONF_PRED
        result = [0.0] * domain.num_actions
        idx = domain.index_for_action("action_default_fallback")
        result[idx] = CONF_PRED
        return result

    def __eq__(self, other):
        return isinstance(other, WorkingPolicy)


def _load_tracker(domain):
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(utils.read_file(tracker_dump))

    tracker = DialogueStateTracker.from_dict(tracker_json.get("sender_id"),
                                             tracker_json.get("events", []),
                                             domain.slots)
    return tracker


def test_fallback_because_of_low_core_confidence(caplog):
    """
    Here a fallback_action shall be triggered because the prediction
    confidences of the other policies are lower then the core_threshold.
    """
    global CONF_PRED
    CONF_PRED = 0.1
    fan = "action_default_fallback"
    fallback_pol = FallbackPolicy(nlu_threshold=0.01, core_threshold=0.3,
                                  fallback_action_name=fan)
    ensemble_pol = SimplePolicyEnsemble([WorkingPolicy(), fallback_pol])

    # load domain
    domain = Domain.load(DOMAIN_PATH)

    # load tracker
    tracker = _load_tracker(domain)

    ensemble_pol.probabilities_using_best_policy(tracker, domain)
    assert "Core confidence" in caplog.text


def test_fallback_not_because_of_low_core_confidence(caplog):
    """
    Here no fallback_action shall be triggered because the prediction
    confidences of the other policies are lower then the core_threshold.
    """
    global CONF_PRED
    CONF_PRED = 1
    fan = "action_default_fallback"
    fallback_pol = FallbackPolicy(nlu_threshold=0.01, core_threshold=0.3,
                                  fallback_action_name=fan)
    ensemble_pol = SimplePolicyEnsemble([WorkingPolicy(), fallback_pol])

    # load domain
    domain = Domain.load(DOMAIN_PATH)

    # load tracker
    tracker = _load_tracker(domain)

    ensemble_pol.probabilities_using_best_policy(tracker, domain)
    assert "Core confidence" not in caplog.text
