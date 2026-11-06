"""fuse() with hand-made views — no model needed."""
import pytest
from fusion import fuse
from model import CLASSES

T = 0.80


def view(label, conf, rest=None):
    """A fake predict() result: `conf` on `label`, the remainder on `rest` (default: the opposite state)."""
    rest = rest or (label.replace('fresh', 'rotten') if 'fresh' in label else label.replace('rotten', 'fresh'))
    probs = {c: 0.0 for c in CLASSES}
    probs[label] = conf
    probs[rest] += round(1 - conf, 4)
    decision = 'REVIEW' if conf < T else ('REJECT' if label.startswith('rotten') else 'PASS')
    return {'label': label, 'confidence': conf, 'decision': decision, 'threshold': T,
            'probabilities': probs, 'heatmap_png': 'x'}


KEY = [view('freshapples', 0.97), view('freshapples', 0.95), view('rottenapples', 0.90)]


def test_key_case_worst_rejects():
    r = fuse(KEY, T, 'worst')
    assert (r['decision'], r['reason'], r['label']) == ('REJECT', 'rotten_view', 'rottenapples')
    assert r['confidence'] == 0.90
    assert r['max_rotten_score'] == pytest.approx(0.90)
    assert r['decided_by'] == 3


def test_key_case_mean_reviews():
    r = fuse(KEY, T, 'mean')
    assert (r['decision'], r['reason'], r['label']) == ('REVIEW', 'mean_low_confidence', 'freshapples')
    assert r['confidence'] == pytest.approx((0.97 + 0.95 + 0.10) / 3, abs=1e-4)
    assert r['probabilities']['rottenapples'] == pytest.approx((0.03 + 0.05 + 0.90) / 3, abs=1e-4)
    assert r['decided_by'] == 3          # the view that disagrees most with "freshapples"


def test_worst_reports_most_confident_rotten_view():
    r = fuse([view('rottenbanana', 0.85), view('freshbanana', 0.99), view('rottenbanana', 0.96)], T)
    assert r['decision'] == 'REJECT' and r['confidence'] == 0.96 and r['decided_by'] == 3
    assert r['fruit_th'] == 'กล้วย' and r['state_th'] == 'เน่า'


def test_worst_reject_beats_fruit_mismatch():
    r = fuse([view('rottenapples', 0.9), view('freshoranges', 0.95)], T)
    assert (r['decision'], r['reason']) == ('REJECT', 'rotten_view')


def test_worst_fruit_mismatch_reviews():
    r = fuse([view('freshapples', 0.95), view('freshoranges', 0.99)], T)
    assert (r['decision'], r['reason']) == ('REVIEW', 'fruit_mismatch')


def test_worst_low_confidence_reviews():
    # a low-confidence rotten view is not a confident REJECT → REVIEW, not REJECT
    r = fuse([view('freshapples', 0.99), view('rottenapples', 0.6)], T)
    assert (r['decision'], r['reason'], r['confidence']) == ('REVIEW', 'low_confidence', 0.6)


def test_worst_all_fresh_passes_with_least_confidence():
    r = fuse([view('freshoranges', 0.99), view('freshoranges', 0.88), view('freshoranges', 0.93)], T)
    assert (r['decision'], r['reason'], r['confidence'], r['decided_by']) == ('PASS', 'all_fresh', 0.88, 2)


def test_mean_pass_and_reject():
    assert fuse([view('freshapples', 0.9), view('freshapples', 0.96)], T, 'mean')['decision'] == 'PASS'
    r = fuse([view('rottenoranges', 0.9), view('rottenoranges', 0.85)], T, 'mean')
    assert (r['decision'], r['reason'], r['decided_by']) == ('REJECT', 'mean_rotten', 1)


@pytest.mark.parametrize('mode', ['worst', 'mean'])
def test_single_view_matches_predict(mode):
    for v in [view('freshapples', 0.95), view('rottenbanana', 0.91), view('freshoranges', 0.5)]:
        r = fuse([v], T, mode)
        assert (r['decision'], r['label'], r['confidence']) == (v['decision'], v['label'], v['confidence'])


def test_output_shape():
    r = fuse(KEY, T)
    assert set(r) == {'mode', 'n_views', 'decision', 'reason', 'label', 'fruit_th', 'state_th', 'confidence',
                      'decided_by', 'threshold', 'max_rotten_score', 'probabilities', 'views'}
    assert r['n_views'] == 3 and [v['view'] for v in r['views']] == [1, 2, 3]
    assert [v['rotten_score'] for v in r['views']] == [0.03, 0.05, 0.9]
    assert r['views'][0]['heatmap_png'] == 'x'            # full predict() result is kept
    assert KEY[0].get('view') is None                     # input views are not mutated


def test_bad_input():
    with pytest.raises(ValueError):
        fuse(KEY, T, 'median')
    with pytest.raises(ValueError):
        fuse([], T)
