import numpy as np
import sklearn
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    label_ranking_average_precision_score,
)


class MyMetrics:
    """Custom metrics for ICD-9 code predictions.

    Code based on HTDC (Ng et al, 2022)"""

    def __init__(self, debug=False):
        self.debug = debug

    def _compute(self, hyps, refs, pred_cutoff=0.5):
        results = {}

        preds = (hyps > pred_cutoff).astype(float)

        has_pos = np.sum(refs, axis=0) != 0
        hyps_for_auc = hyps[:, has_pos]
        refs_for_auc = refs[:, has_pos]

        results["f1_macro"] = f1_score(
            refs.tolist(), preds.tolist(), average="macro", zero_division=0
        )
        results["f1_micro"] = f1_score(
            refs.tolist(), preds.tolist(), average="micro", zero_division=0
        )
        results["f1_by_class"] = f1_score(
            refs.tolist(), preds.tolist(), average=None, zero_division=0
        )
        if not self.debug:
            results["auc_macro"] = roc_auc_score(
                refs_for_auc.tolist(), hyps_for_auc.tolist(), average="macro"
            )
            results["auc_micro"] = roc_auc_score(
                refs_for_auc.tolist(), hyps_for_auc.tolist(), average="micro"
            )
            results["auc_by_class"] = roc_auc_score(
                refs_for_auc.tolist(), hyps_for_auc.tolist(), average=None
            )

            results["p_5"] = self.precision_at_k(hyps, refs)

            results["LRAP"] = label_ranking_average_precision_score(
                refs.tolist(), hyps.tolist()
            )
        results["metrics_sample_size"] = len(hyps)

        return results

    def from_torch(self, hyps, refs):
        return self._compute(hyps.cpu().detach().numpy(), refs.cpu().detach().numpy())

    def from_numpy(self, hyps, refs, pred_cutoff=0.5):
        return self._compute(hyps, refs, pred_cutoff=pred_cutoff)

    def precision_at_k(self, hyps, refs, k=5):
        """
        Adapted from: Vu et al (2020). https://github.com/aehrc/LAAT
        """
        sorted_pred = np.argsort(hyps)[:, ::-1]

        topk = sorted_pred[:, :k]
        numerator = 0
        denominator = 0
        vals = []
        for i, tk in enumerate(topk):
            if len(tk) > 0:
                num_true_in_top_k = np.sum(refs[i, tk])
                denom = len(tk)

                numerator += num_true_in_top_k
                denominator += denom
                vals.append(num_true_in_top_k / float(denom))

        return np.mean(vals)

    def get_optimal_microf1_threshold_v2(self, hyps, refs):
        eps = 1e-8
        hyps_all = hyps.ravel()
        refs_all = refs.ravel()

        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
            refs_all, hyps_all
        )
        f1 = 2 * recall * precision / (recall + precision + eps)
        best_threshold = thresholds[np.argmax(f1)]
        best_f1 = np.max(f1)

        return best_threshold
