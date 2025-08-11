import numpy as np
import pandas as pd
import pyannote.audio
print(f"pyannote.audio version: {pyannote.audio.__version__}")
from whisperx.diarizen_pipeline import DiarizenPipelineWrapper
from typing import Optional, Union
import torch
from pathlib import Path

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.types import TranscriptionResult, AlignedTranscriptionResult


class DiarizationPipeline:
    def __init__(
        self,
        model_name="BUT-FIT/diarizen-wavlm-base-s80-md",
        rttm_out_dir: Optional[str] = None,
        **kwargs,
    ):
        self.model = DiarizenPipelineWrapper(
            model_name=model_name,
            rttm_out_dir=rttm_out_dir
        )

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> Union[tuple[pd.DataFrame, Optional[dict[str, list[float]]]], pd.DataFrame]:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Path to audio file or audio array
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            return_embeddings: Whether to return speaker embeddings

        Returns:
            If return_embeddings is True:
                Tuple of (diarization dataframe, speaker embeddings dictionary)
            Otherwise:
                Just the diarization dataframe
        """
        if isinstance(audio, str):
            audio_path = audio
            audio_input = load_audio(audio_path)
        elif isinstance(audio, np.ndarray):
            audio_input = audio
            audio_path = "audio_from_array"
        else:
            raise TypeError("audio must be a path to an audio file or a numpy array")

        sess_name = Path(audio_path).stem

        diarization = self.model(
            audio_input,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            sess_name=sess_name,
        )

        diarize_df = pd.DataFrame(diarization.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        # Ensure JSON-safe speaker labels (avoid numpy int64). Use string labels like "SPEAKER_0".
        try:
            diarize_df['speaker'] = diarize_df['speaker'].apply(lambda s: f"SPEAKER_{int(s)}")
        except Exception:
            diarize_df['speaker'] = diarize_df['speaker'].astype(str)

        if return_embeddings:
            return diarize_df, None
        else:
            return diarize_df


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
) -> Union[AlignedTranscriptionResult, TranscriptionResult]:
    """
    Assign speakers to words and segments in the transcript.

    Args:
        diarize_df: Diarization dataframe from DiarizationPipeline
        transcript_result: Transcription result to augment with speaker labels
        speaker_embeddings: Optional dictionary mapping speaker IDs to embedding vectors
        fill_nearest: If True, assign speakers even when there's no direct time overlap

    Returns:
        Updated transcript_result with speaker assignments and optionally embeddings
    """
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
        # remove no hit, otherwise we look for closest (even negative intersection...)
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg["speaker"] = speaker
        
        # assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'], word['start'])
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                        word["speaker"] = speaker

    # Add speaker embeddings to the result if provided
    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings

    return transcript_result


class Segment:
    def __init__(self, start:int, end:int, speaker:Optional[str]=None):
        self.start = start
        self.end = end
        self.speaker = speaker
