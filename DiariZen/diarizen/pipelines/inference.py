# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

import toml
import yaml
import numpy as np
import torch
import torchaudio

from scipy.ndimage import median_filter

from huggingface_hub import snapshot_download, hf_hub_download
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio.pipelines.clustering import Clustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio, Pipeline
from pyannote.audio.utils.signal import Binarize
from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.audio.core.inference import Inference

from diarizen.pipelines.utils import scp2path


class DiariZenPipeline(SpeakerDiarizationPipeline):
    def __init__(
        self, 
        diarizen_hub,
        embedding_model,
        config_parse: Optional[Dict[str, Any]] = None,
        rttm_out_dir: Optional[str] = None,
    ):
        config_path = Path(diarizen_hub / "config.toml")
        config = toml.load(config_path.as_posix())

        if config_parse is not None:
            print('Overriding with parsed config.')
            config["inference"]["args"] = config_parse["inference"]["args"]
            config["clustering"]["args"] = config_parse["clustering"]["args"]
       
        inference_config = config["inference"]["args"]
        # clustering args may be partially missing in overrides; use safe defaults
        clustering_args = config.get("clustering", {}).get("args", {})
        
        print(f'Loaded configuration: {config}')

        # Save config as yaml for pyannote
        yaml_path = Path(diarizen_hub) / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        # IMPORTANT: Do not call SpeakerDiarization.__init__ as it would try to
        # download a gated pyannote/segmentation model. We initialize the base
        # Pipeline directly and set up only the components we actually use
        # (embedding + clustering). Segmentation is performed by DiariZen below.
        Pipeline.__init__(self)

        # mirror key attributes used by SpeakerDiarization
        self.embedding = embedding_model
        self.embedding_batch_size = inference_config["batch_size"]
        self.embedding_exclude_overlap = True
        method = clustering_args.get("method", "VBxClustering")
        self.klustering = method
        # device selection: prefer CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Accuracy preference: disable TF32 on CUDA
        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass

        # set up speaker embeddings
        self._embedding = PretrainedSpeakerEmbedding(
            self.embedding, device=self.device, use_auth_token=None
        )
        self._audio = Audio(sample_rate=self._embedding.sample_rate, mono="downmix")
        metric = self._embedding.metric

        # set up clustering
        Klustering = Clustering[self.klustering]
        self.clustering = Klustering.value(metric=metric)

        self.apply_median_filtering = inference_config["apply_median_filtering"]
        self.min_speakers = clustering_args.get("min_speakers", 1)
        self.max_speakers = clustering_args.get("max_speakers", 20)

        if method == "AgglomerativeClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": clustering_args.get("min_cluster_size", 2),
                    "threshold": clustering_args.get("ahc_threshold", 0.6),
                }
            }
        elif method == "VBxClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "ahc_criterion": clustering_args.get("ahc_criterion", "distance"),
                    "ahc_threshold": clustering_args.get("ahc_threshold", 0.6),
                    "Fa": clustering_args.get("Fa", 0.07),
                    "Fb": clustering_args.get("Fb", 0.8),
                }
            }
            self.clustering.plda_dir = str(Path(diarizen_hub / "plda"))
            self.clustering.lda_dim = clustering_args.get("lda_dim", 128)
            self.clustering.maxIters = clustering_args.get("max_iters", 20)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        self.instantiate(self.PIPELINE_PARAMS)

        # ---------- DiariZen EEND model init ----------
        def _import_string(path: str):
            module, cls = path.rsplit('.', 1)
            mod = __import__(module, fromlist=[cls])
            return getattr(mod, cls)

        model_cfg = config["model"]
        ModelClass = _import_string(model_cfg["path"])
        self.eend_model = ModelClass(**model_cfg["args"])  # subclass of pyannote Model

        # load checkpoint(s)
        root_bin = Path(diarizen_hub) / "pytorch_model.bin"
        try:
            if root_bin.is_file():
                state = torch.load(root_bin.as_posix(), map_location="cpu")
                self.eend_model.load_state_dict(state, strict=False)
            else:
                # try averaging checkpoints if available
                from diarizen.ckpt_utils import average_ckpt
                self.eend_model = average_ckpt(diarizen_hub, self.eend_model, wavlm_only=False)
        except Exception as e:
            print(f"Warning: failed to load DiariZen checkpoint: {e}")

        self.eend_model.to(self.device).eval()

        # create inference helper to handle sliding windows and overlap-add
        self._dz_inference = Inference(
            self.eend_model,
            window="sliding",
            duration=inference_config["seg_duration"],
            step=inference_config["segmentation_step"],
            batch_size=inference_config["batch_size"],
            device=self.device,
            skip_aggregation=False,
        )

        if rttm_out_dir is not None:
            os.makedirs(rttm_out_dir, exist_ok=True)
        self.rttm_out_dir = rttm_out_dir

        # Segmentation is powered by DiariZen EEND and outputs powerset labels
        # directly; no assertion on pyannote segmentation model needed here.

    # override segmentation to avoid pyannote's _segmentation
    def get_segmentations(self, file, hook=None, soft=False):
        return self._dz_inference(file)

    @classmethod
    def from_pretrained(
        cls, 
        repo_id: str, 
        cache_dir: str = None,
        rttm_out_dir: str = None,
        config_parse: Optional[Dict[str, Any]] = None,
    ) -> "DiariZenPipeline":
        diarizen_hub = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

        embedding_model = hf_hub_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            filename="pytorch_model.bin",
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

        return cls(
            diarizen_hub=Path(diarizen_hub).expanduser().absolute(),
            embedding_model=embedding_model,
            rttm_out_dir=rttm_out_dir,
            config_parse=config_parse,
        )

    def __call__(self, in_wav, sess_name=None):
        # Accept path/ProtocolFile or raw numpy array
        if isinstance(in_wav, (str, ProtocolFile)):
            in_wav = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav['audio']
            print('Extracting segmentations.')
            waveform, sample_rate = torchaudio.load(in_wav)
            # force mono: use first channel
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = waveform[:1, :]
        elif isinstance(in_wav, np.ndarray):
            print('Extracting segmentations (ndarray input).')
            arr = in_wav
            if arr.ndim == 1:
                arr = arr[None, :]
            elif arr.ndim == 2 and arr.shape[0] > 1:
                arr = arr[:1, :]
            waveform = torch.from_numpy(arr.astype(np.float32))
            # WhisperX uses 16 kHz throughout; fallback to 16000 for array input
            sample_rate = 16000
        else:
            raise TypeError("input must be either a str, ProtocolFile, or numpy.ndarray")

        # ensure shape [1, T]
        if waveform.dim() == 2 and waveform.size(0) == 1:
            pass
        elif waveform.dim() == 1:
            waveform = waveform[None, :]
        else:
            waveform = waveform[:1, :]
        segmentations = self.get_segmentations({"waveform": waveform, "sample_rate": sample_rate}, soft=False)

        if self.apply_median_filtering:
            segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')

        # binarize segmentation
        binarized_segmentations = segmentations     # powerset

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            binarized_segmentations,
            self.eend_model._receptive_field,
            warm_up=(0.0, 0.0),
        )

        print("Extracting Embeddings.")
        embeddings = self.get_embeddings(
            {"waveform": waveform, "sample_rate": sample_rate},
            binarized_segmentations,
            exclude_overlap=self.embedding_exclude_overlap,
        )

        # shape: (num_chunks, local_num_speakers, dimension)
        print("Clustering.")
        hard_clusters, _, _ = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            min_clusters=self.min_speakers,  
            max_clusters=self.max_speakers
        )

        # during counting, we could possibly overcount the number of instantaneous
        # speakers due to segmentation errors, so we cap the maximum instantaneous number
        # of speakers by the `max_speakers` value
        count.data = np.minimum(count.data, self.max_speakers).astype(np.int8)

        # keep track of inactive speakers
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)

        # reconstruct discrete diarization from raw hard clusters
        hard_clusters[inactive_speakers] = -2
        discrete_diarization, _ = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )

        # convert to annotation
        to_annotation = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=0.0,
            min_duration_off=0.0
        )
        result = to_annotation(discrete_diarization)
        result.uri = sess_name
        
        if self.rttm_out_dir is not None:
            assert sess_name is not None
            rttm_out = os.path.join(self.rttm_out_dir, sess_name + ".rttm")
            with open(rttm_out, "w") as f:
                f.write(result.to_rttm())
        return result
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "This script performs diarization using DiariZen pipeline ",
        add_help=True,
        usage="%(prog)s [options]",
    )

    # Required paths
    parser.add_argument(
        "--in_wav_scp",
        type=str,
        required=True,
        help="Path to wav.scp."
    )
    parser.add_argument(
        "--diarizen_hub",
        type=str,
        required=True,
        help="Path to DiariZen model hub directory."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Path to pretrained embedding model."
    )

    # inference parameters
    parser.add_argument(
        "--seg_duration",
        type=int,
        default=16,
        help="Segment duration in seconds.",
    )
    parser.add_argument(
        "--segmentation_step",
        type=float,
        default=0.1,
        help="Shifting ratio during segmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Input batch size for inference.",
    )
    parser.add_argument(
        "--apply_median_filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply median filtering to segmentation output.",
    )

    # clustering parameters
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="VBxClustering",
        choices=["VBxClustering", "AgglomerativeClustering"],
        help="Clustering method to use.",
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=1,
        help="Minimum number of speakers.",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=20,
        help="Maximum number of speakers.",
    )
    parser.add_argument(
        "--ahc_criterion",
        type=str,
        default="distance",
        help="AHC criterion (for VBx).",
    )
    parser.add_argument(
        "--ahc_threshold",
        type=float,
        default=0.6,
        help="AHC threshold.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=13,
        help="Minimum cluster size (for AHC).",
    )
    parser.add_argument(
        "--Fa",
        type=float,
        default=0.07,
        help="VBx Fa parameter.",
    )
    parser.add_argument(
        "--Fb",
        type=float,
        default=0.8,
        help="VBx Fb parameter.",
    )
    parser.add_argument(
        "--lda_dim",
        type=int,
        default=128,
        help="VBx LDA dimension.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20,
        help="VBx maximum iterations.",
    )

    # Output
    parser.add_argument(
        "--rttm_out_dir",
        type=str,
        default=None,
        required=False,
        help="Path to output folder.",
    )

    args = parser.parse_args()
    print(args)

    inference_config = {
        "seg_duration": args.seg_duration,
        "segmentation_step": args.segmentation_step,
        "batch_size": args.batch_size,
        "apply_median_filtering": args.apply_median_filtering
    }

    clustering_config = {
        "method": args.clustering_method,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers
    }
    if args.clustering_method == "AgglomerativeClustering":
        clustering_config.update({
            "ahc_threshold": args.ahc_threshold,
            "min_cluster_size": args.min_cluster_size
        })
    elif args.clustering_method == "VBxClustering":
        clustering_config.update({
            "ahc_criterion": args.ahc_criterion,
            "ahc_threshold": args.ahc_threshold,
            "Fa": args.Fa,
            "Fb": args.Fb,
            "lda_dim": args.lda_dim,
            "max_iters": args.max_iters
        })
    else:
        raise ValueError(f"Unsupported clustering method: {args.clustering_method}")

    config_parse = {
        "inference": {"args": inference_config},
        "clustering": {"args": clustering_config}
    }

    diarizen_pipeline = DiariZenPipeline(
        diarizen_hub=Path(args.diarizen_hub),
        embedding_model=args.embedding_model,
        config_parse=config_parse,
        rttm_out_dir=args.rttm_out_dir
    )

    audio_f = scp2path(args.in_wav_scp)
    for audio_file in audio_f:
        sess_name = Path(audio_file).stem.split('.')[0]
        print(f'Prosessing: {sess_name}')
        diarizen_pipeline(audio_file, sess_name=sess_name)
