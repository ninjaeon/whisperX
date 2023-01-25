import numpy as np
import pandas as pd

def assign_word_speakers(diarize_df, result_segments, fill_nearest=False):
 
    for seg in result_segments:
        wdf = seg['word-segments']
        if len(wdf['start'].dropna()) == 0:
            wdf['start'] = seg['start']
            wdf['end'] = seg['end']
        speakers = []
        for wdx, wrow in wdf.iterrows():
            if not np.isnan(wrow['start']):
                diarize_df['intersection'] = np.minimum(diarize_df['end'], wrow['end']) - np.maximum(diarize_df['start'], wrow['start'])
                diarize_df['union'] = np.maximum(diarize_df['end'], wrow['end']) - np.minimum(diarize_df['start'], wrow['start'])
                # remove no hit
                if not fill_nearest:
                    dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                else:
                    dia_tmp = diarize_df
                if len(dia_tmp) == 0:
                    speaker = None
                else:
                    speaker = dia_tmp.sort_values("intersection", ascending=False).iloc[0][2]
            else:
                speaker = None
            speakers.append(speaker)
        seg['word-segments']['speaker'] = speakers
        seg["speaker"] = pd.Series(speakers).value_counts().index[0]

    # create word level segments for .srt
    word_seg = []
    for seg in result_segments:
        wseg = pd.DataFrame(seg["word-segments"])
        for wdx, wrow in wseg.iterrows():
            if wrow["start"] is not None:
                speaker = wrow['speaker']
                if speaker is None or speaker == np.nan:
                    speaker = "UNKNOWN"
                word_seg.append(
                    {
                        "start": wrow["start"],
                        "end": wrow["end"],
                        "text": f"[{speaker}]: " + seg["text"][int(wrow["segment-text-start"]):int(wrow["segment-text-end"])]
                    }
                )

    # TODO: create segments but split words on new speaker

    return result_segments, word_seg

class Segment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
