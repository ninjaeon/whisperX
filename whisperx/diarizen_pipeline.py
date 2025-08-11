from diarizen.pipelines.inference import DiariZenPipeline
from typing import Union
import numpy as np

class DiarizenPipelineWrapper:
    def __init__(self, model_name: str, rttm_out_dir: str = None):
        # Override default inference params to reduce VRAM and improve speed
        config_parse = {
            "inference": {
                "args": {
                    # Match training chunk size for better quality/speed tradeoff
                    "seg_duration": 16,
                    # More overlap improves accuracy
                    "segmentation_step": 0.25,
                    # Smaller batch for VRAM stability
                    "batch_size": 6,
                    "apply_median_filtering": True,
                }
            },
            "clustering": {
                "args": {
                    # Explicit method to avoid KeyError in pipeline parsing
                    "method": "VBxClustering",
                    # Reasonable defaults; can be overridden by caller later
                    "min_speakers": 1,
                    "max_speakers": 10,
                    # Make clustering stricter to reduce over-segmentation
                    "ahc_criterion": "distance",
                    "ahc_threshold": 0.9,
                }
            }
        }
        self.pipeline = DiariZenPipeline.from_pretrained(
            model_name,
            rttm_out_dir=rttm_out_dir,
            config_parse=config_parse,
        )

    def __call__(self, audio: Union[str, np.ndarray], min_speakers: int, max_speakers: int, sess_name: str = None):
        """
        Diarizen does not support min_speakers and max_speakers.
        """
        # Best-effort: inject speaker bounds into pipeline before run
        try:
            if min_speakers is not None:
                self.pipeline.min_speakers = int(min_speakers)
            if max_speakers is None:
                max_speakers = 10
            self.pipeline.max_speakers = int(max_speakers)
            # If VBx is used, also tighten AHC threshold when user specifies low max_speakers
            if hasattr(self.pipeline, "klustering") and self.pipeline.klustering == "VBxClustering":
                if max_speakers is not None and max_speakers <= 6 and hasattr(self.pipeline, "clustering"):
                    try:
                        # Raise threshold slightly to merge more clusters
                        self.pipeline.clustering.ahc_threshold = max(0.75, getattr(self.pipeline.clustering, "ahc_threshold", 0.8))
                    except Exception:
                        pass
        except Exception:
            pass

        diarization_result = self.pipeline(audio, sess_name=sess_name)
        return diarization_result
