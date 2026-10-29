# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
def binary_metrics(rows):
    counts = {k: sum((r[k] for r in rows)) for k in ('TP', 'FP', 'FN', 'TN')}
    tp, fp, fn, tn = (counts[k] for k in ('TP', 'FP', 'FN', 'TN'))

    def ratio(a, b):
        return a / b * 100.0 if b else float('nan')
    ciou = ratio(tp, tp + fp + fn)
    biou = ratio(tn, tn + fp + fn)
    return dict(mIoU=(ciou + biou) / 2, CrackIoU=ciou, CrackF1=ratio(2 * tp, 2 * tp + fp + fn), Precision=ratio(tp, tp + fp), Recall=ratio(tp, tp + fn), aAcc=ratio(tp + tn, tp + fp + fn + tn), counts=counts)
