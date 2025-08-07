from diarizen.pipelines.inference import DiariZenPipeline
from typing import Union
import numpy as np

class DiarizenPipelineWrapper:
    def __init__(self, model_name: str, rttm_out_dir: str = None):
        self.pipeline = DiariZenPipeline.from_pretrained(
            model_name,
            rttm_out_dir=rttm_out_dir
        )

    def __call__(self, audio: Union[str, np.ndarray], min_speakers: int, max_speakers: int, sess_name: str = None):
        """
        Diarizen does not support min_speakers and max_speakers.
        """
        diarization_result = self.pipeline(audio, sess_name=sess_name)
        return diarization_result
