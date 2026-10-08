import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('review_summary', Path(__file__).parents[1] / 'scripts/summarize_quality_review.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(pred='N', human='C', revealed=False):
    data = {'clips': [{'sha256': 'abc', 'report': {'context_dimensions': {'hand_visibility': ''},
        'windows': [{'start_s': 0, 'end_s': 8, 'rows': {'hand_visibility': {'status': pred}}}]}}]}
    labels = {'reviews': [{'source_sha256': 'abc', 'start_s': 0, 'end_s': 8,
        'ratings': {'hand_visibility': {'status': human, 'model_revealed_before_rating': revealed}}}]}
    return data, labels


def test_missed_concern_counts_as_false_negative():
    result = module.summarize(*fixture())['dimensions']['hand_visibility']
    assert result['missed_concerns'] == 1
    assert result['resolved_concern_recall'] == 0


def test_abstention_is_unresolved_not_success():
    result = module.summarize(*fixture(pred='U'))['dimensions']['hand_visibility']
    assert result['model_unknown_or_missing'] == 1
    assert result['concern_recall_on_model_definite'] is None
    assert result['resolved_concern_recall'] == 0


def test_revealed_ratings_excluded_by_default():
    data, human = fixture(revealed=True)
    assert not module.summarize(data, human)['dimensions']
    assert module.summarize(data, human, include_revealed=True)['dimensions']


def test_unreviewed_and_uncertain_are_not_ground_truth():
    assert not module.summarize(*fixture(human='U'))['dimensions']
    data, human = fixture()
    human['reviews'][0]['ratings'] = {}
    assert not module.summarize(data, human)['dimensions']


def test_unknown_source_and_duplicate_labels_rejected():
    data, human = fixture()
    human['reviews'][0]['source_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='hash'):
        module.summarize(data, human)
    data, human = fixture()
    human['reviews'] *= 2
    with pytest.raises(ValueError, match='Duplicate'):
        module.summarize(data, human)
