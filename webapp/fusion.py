"""Combine several predict() results for the SAME fruit (one per camera) into one verdict.

The model is unchanged: predict() runs once per photo, then fuse() decides.
- worst: the most cautious view wins — one confidently rotten view rejects the fruit.
- mean:  average the class probabilities across views, then apply the usual threshold.
"""
from model import CLASSES, FRUIT_TH, ROTTEN

MODES = ('worst', 'mean')

# reason codes (the page shows a Thai sentence for each)
#   rotten_view         worst: at least one view is a confident REJECT
#   fruit_mismatch      worst: views disagree on the fruit type
#   low_confidence      worst: at least one view is below the threshold
#   all_fresh           worst: every view is a confident PASS
#   mean_rotten / mean_fresh / mean_low_confidence   mean: verdict of the averaged probabilities
# decided_by: number of the view that decided (worst) or best explains (mean) the verdict


def _fruit(label):
    return label.replace('fresh', '').replace('rotten', '')


def _rotten_score(probs):
    return sum(probs[CLASSES[i]] for i in ROTTEN)


def fuse(views, threshold, mode='worst'):
    if mode not in MODES:
        raise ValueError(f'unknown fusion mode: {mode}')
    if not views:
        raise ValueError('need at least one view')

    views = [{**v, 'view': n, 'rotten_score': round(_rotten_score(v['probabilities']), 4)}
             for n, v in enumerate(views, start=1)]
    mean_probs = {c: sum(v['probabilities'][c] for v in views) / len(views) for c in CLASSES}

    if mode == 'worst':
        rejects = [v for v in views if v['decision'] == 'REJECT']
        if rejects:
            decision, reason = 'REJECT', 'rotten_view'
            shown = max(rejects, key=lambda v: v['confidence'])       # most confident rotten view
        else:
            shown = min(views, key=lambda v: v['confidence'])          # least confident view
            if len({_fruit(v['label']) for v in views}) > 1:
                decision, reason = 'REVIEW', 'fruit_mismatch'
            elif any(v['decision'] == 'REVIEW' for v in views):
                decision, reason = 'REVIEW', 'low_confidence'
            else:
                decision, reason = 'PASS', 'all_fresh'
        label, conf = shown['label'], shown['confidence']
        decided_by = shown['view']
    else:
        idx = max(range(len(CLASSES)), key=lambda i: mean_probs[CLASSES[i]])
        label, conf = CLASSES[idx], mean_probs[CLASSES[idx]]
        if conf < threshold:
            decision, reason = 'REVIEW', 'mean_low_confidence'
        elif idx in ROTTEN:
            decision, reason = 'REJECT', 'mean_rotten'
        else:
            decision, reason = 'PASS', 'mean_fresh'
        # the camera that best explains the verdict: most sure of the label, or (REVIEW) the one that disagrees most
        pick = min if decision == 'REVIEW' else max
        decided_by = pick(views, key=lambda v: v['probabilities'][label])['view']

    return {
        'mode': mode,
        'n_views': len(views),
        'decision': decision,
        'reason': reason,
        'label': label,
        'fruit_th': FRUIT_TH[_fruit(label)],
        'state_th': 'เน่า' if CLASSES.index(label) in ROTTEN else 'สด',
        'confidence': round(conf, 4),
        'decided_by': decided_by,
        'threshold': threshold,
        'max_rotten_score': max(v['rotten_score'] for v in views),
        'probabilities': {c: round(p, 4) for c, p in mean_probs.items()},
        'views': views,
    }
