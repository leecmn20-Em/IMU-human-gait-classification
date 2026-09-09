import numpy as np

class OnlinePostProcessor:
    transition_classes = {2,3,5}
    
    def __init__(self, causality, k=3, tau=0.5, tau_trans=0.65, force_tau=0.9):
        assert 0.0 <= tau <= 1.0
        assert 0.0 <= tau_trans <= 1.0
        assert 0.0 <= force_tau <= 1.0
        assert force_tau >= tau and force_tau >= tau_trans
        
        self.causality = causality.astype(bool)
        self.k = k
        self.tau = tau
        self.force_tau = force_tau
        self.tau_trans = tau_trans

        self.state = None
        self.cand = None
        self.cand_count = 0

    def step(self, prob_t):
        prob_t = np.asarray(prob_t).reshape(-1)

        raw_cls = int(prob_t.argmax())
        pmax = float(prob_t[raw_cls])

        if self.state is None:
            self.state = raw_cls
            return self.state
        
        threshold = self.tau_trans if self.cand in self.transition_classes else self.tau
        if pmax < threshold:
            return self.state

        # causality check
        masked = prob_t.copy()
        mask_allowed = self.causality[self.state]
        masked[~mask_allowed] = 0.0

        new_cls = int(masked.argmax())
        new_p = float(masked[new_cls])

        if new_p <= 0.0:
            return self.state

        # 강제 전환
        if new_cls != self.state and new_p >= self.force_tau:
            self.state = new_cls
            self.cand = None
            self.cand_count = 0
            return self.state

        # 디바운스
        if new_cls == self.state:
            self.cand = None
            self.cand_count = 0
            return self.state

        if self.cand == new_cls:
            self.cand_count += 1
        else:
            self.cand = new_cls
            self.cand_count = 1

        if self.cand_count >= self.k:
            self.state = self.cand
            self.cand = None
            self.cand_count = 0

        return self.state
