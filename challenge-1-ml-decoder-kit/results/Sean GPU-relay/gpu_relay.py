"""Single-syndrome CUDA-Q QEC relay-BP adapter."""
import numpy as np

from decoders import RELAY_SETTINGS


class GpuRelay:
    def __init__(self, H, priors, stop_nconv=5, **overrides):
        self.H = H.toarray(order="C").astype(np.uint8, copy=False)
        s = RELAY_SETTINGS
        self.options = dict(bp_method=3, composition=1, use_sparsity=True,
                            output="errors", use_osd=False, proc_float="fp64",
                            bp_seed=8, bp_batch_size=1, scale_factor=s["alpha"],
                            gamma0=s["gamma0"], gamma_dist=list(s["gamma_dist_interval"]),
                            max_iterations=s["set_max_iter"],
                            srelay_config=dict(pre_iter=s["pre_iter"], num_sets=s["num_sets"],
                                               stopping_criterion="NConv", stop_nconv=stop_nconv),
                            opt_results={"num_iter": True}) | overrides
        self.last_iterations = -1
        self.set_priors(priors)

    def set_priors(self, priors):
        """Rebuild the native decoder because QEC 0.9 exposes priors at construction."""
        import cudaq_qec as qec

        rates = np.array(priors, dtype=np.float64, order="C", copy=True)
        if rates.shape != (self.H.shape[1],):
            raise ValueError("expected one prior per column")
        options = self.options | {"error_rate_vec": rates}
        self.decoder = qec.get_decoder("nv-qldpc-decoder", self.H, **options)
        self.options = options

    def decode(self, syndrome):
        result = self.decoder.decode(np.ascontiguousarray(syndrome, dtype=np.uint8))
        self.last_iterations = result.opt_results.get("num_iter", -1)
        error = np.asarray(result.result, dtype=np.uint8)
        if error.shape != (self.H.shape[1],):
            raise ValueError("decoder returned an unexpected correction shape")
        return error
