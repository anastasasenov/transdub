#!/usr/bin/env python3

# transdub.py — Cross-platform system-audio transcription, translation, and dubbing.
#
#  pip install "sounddevice>=0.4.6" "numpy>=1.22" "scipy>=1.8" \
#      "openai-whisper>=20230314" "deep-translator>=1.11.4" \
#      "gTTS>=2.3.2" "pydub>=0.25.1"
#
#  Install FFmpeg and ensure ffmpeg is on PATH.


from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
DEFAULT_MODEL = "base"

LOG = logging.getLogger("transdub")


def configure_logging(
    verbose: bool = False
    ) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

def require_dependencies():
    missing = []
    modules: dict[str, Any] = {}

    imports = {
        "sounddevice": "sounddevice",
        "numpy": "numpy",
        "scipy.io.wavfile": "scipy.io.wavfile",
        "whisper": "whisper",
        "deep_translator": "deep_translator",
        "gtts": "gtts",
        "pydub": "pydub",
    }
    for module_name, import_name in imports.items():
        try:
            if module_name == "scipy.io.wavfile":
                from scipy.io import wavfile
                modules["wavfile"] = wavfile
            elif module_name == "deep_translator":
                from deep_translator import GoogleTranslator
                modules["GoogleTranslator"] = GoogleTranslator
            elif module_name == "gtts":
                from gtts import gTTS
                modules["gTTS"] = gTTS
            else:
                modules[module_name] = __import__(import_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        raise RuntimeError(
            "Missing Python dependencies: "
            + ", ".join(missing)
            + "\nInstall them with:\n"
              "  pip install sounddevice numpy scipy openai-whisper "
              "deep-translator gTTS pydub"
        )
    return modules
    
def check_ffmpeg() -> None:
    from shutil import which

    if which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg was not found on PATH. Install FFmpeg and restart the terminal. "
            "It is required by pydub and Whisper."
        )


def query_devices(
    sd
    ) -> tuple[list[dict[str, Any]], Optional[int]]:
    try:
        devices = list(sd.query_devices())
    except Exception as exc:
        raise RuntimeError(f"Could not query audio devices: {exc}") from exc

    system = platform.system().lower()
    hostapis = []
    try:
        hostapis = list(sd.query_hostapis())
    except Exception:
        pass

    def host_name(device: dict[str, Any]) -> str:
        try:
            return str(hostapis[int(device.get("hostapi", -1))].get("name", ""))
        except (IndexError, TypeError, ValueError):
            return ""

    candidates: list[tuple[int, int]] = []
    for index, dev in enumerate(devices):
        name = str(dev.get("name", ""))
        lname = name.lower()
        hname = host_name(dev).lower()
        input_channels = int(dev.get("max_input_channels", 0) or 0)
        output_channels = int(dev.get("max_output_channels", 0) or 0)
        score = 0

        if system == "windows":
            if "wasapi" in hname:
                score += 10
            if "loopback" in lname:
                score += 100
            if "wasapi" in hname and output_channels > 0:
                score += 20
            if score and (input_channels > 0 or output_channels > 0):
                candidates.append((score, index))

        elif system == "linux":
            if input_channels > 0 and ("monitor" in lname or "monitor" in hname):
                score = 100 if "monitor" in lname else 50
                candidates.append((score, index))

        elif system == "darwin":
            if input_channels > 0 and "blackhole" in lname:
                candidates.append((100, index))

    selected = max(candidates, default=(0, -1))[1]
    return devices, selected if selected >= 0 else None

def print_devices(
    sd, 
    devices: list[dict[str, Any]], 
    selected: Optional[int]
    ) -> None:
    try:
        hostapis = list(sd.query_hostapis())
    except Exception:
        hostapis = []

    print("\nAudio devices:")
    if not devices:
        print("  (No devices returned by PortAudio.)")
        return

    for index, dev in enumerate(devices):
        try:
            host = hostapis[int(dev.get("hostapi", -1))]["name"]
        except (IndexError, TypeError, ValueError):
            host = "unknown host API"
        marker = "  <== auto-selected loopback candidate" if index == selected else ""
        print(
            f"  [{index:>3}] {dev.get('name', 'Unnamed')} | "
            f"host={host} | inputs={dev.get('max_input_channels', 0)} | "
            f"outputs={dev.get('max_output_channels', 0)}{marker}"
        )

    if selected is None:
        print("\nNo automatic loopback device was recognized.")
        print("Use --device INDEX to choose an input explicitly, or configure loopback.")

def select_capture_device(
    sd,
    devices,
    auto_index,
    explicit_index
    ) -> None:
    if explicit_index is not None:
        if explicit_index < 0 or explicit_index >= len(devices):
            raise RuntimeError(
                f"Device index {explicit_index} is out of range; "
                f"valid indices are 0 through {max(0, len(devices) - 1)}."
            )
        device = devices[explicit_index]
        if int(device.get("max_input_channels", 0) or 0) < 1:
            hostapis = list(sd.query_hostapis())
            try:
                host = str(hostapis[int(device.get("hostapi", -1))]["name"]).lower()
            except (IndexError, TypeError, ValueError):
                host = ""
            if not (platform.system().lower() == "windows" and "wasapi" in host
                    and int(device.get("max_output_channels", 0) or 0) > 0):
                raise RuntimeError(
                    f"Device {explicit_index} has no input channels and cannot be "
                    "used as a normal capture input."
                )
        return explicit_index, False

    if auto_index is not None:
        return auto_index, False

    LOG.warning(
        "No loopback device was detected. Falling back to the default input device; "
        "this may record microphone/ambient audio instead of system output."
    )
    default_input = getattr(sd.default, "device", (None, None))
    if isinstance(default_input, (tuple, list)) and default_input:
        default_input = default_input[0]
    if default_input is None or default_input == -1:
        raise RuntimeError(
            "No loopback device or default input device is available. "
            "Connect/configure an input device or pass --device INDEX."
        )
    return int(default_input), True
    
def capture_audio(
    sd,
    np,
    wavfile,
    device_index: int,
    duration: int,
    wav_path: str,
    ambient_fallback: bool
    ) -> None:
    LOG.info("Capturing audio for %d seconds from device %d...", duration, device_index)

    extra_settings = None
    if platform.system().lower() == "windows":
        try:
            hostapis = list(sd.query_hostapis())
            dev = sd.query_devices(device_index)
            host = str(hostapis[int(dev["hostapi"])]["name"]).lower()
            if "wasapi" in host and int(dev.get("max_input_channels", 0) or 0) == 0:
                extra_settings = sd.WasapiSettings(loopback=True)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            extra_settings = None

    try:
        device_info = sd.query_devices(device_index)
        max_in = int(device_info.get("max_input_channels", 0) or 0)
        if max_in > 0:
            input_channels = 1
        else:
            input_channels = 2
        kwargs = dict(
            samplerate=SAMPLE_RATE,
            channels=input_channels,
            dtype=DTYPE,
            device=device_index,
            blocking=True,
        )
        if extra_settings is not None:
            kwargs["extra_settings"] = extra_settings

        frames = int(SAMPLE_RATE * duration)
        recording = sd.rec(frames, **{k: v for k, v in kwargs.items() if k != "blocking"},
                           blocking=True)
        if recording is None or len(recording) == 0:
            raise RuntimeError("The capture device returned no audio frames.")
        if recording.ndim == 2 and recording.shape[1] > 1:
            mono = np.mean(recording.astype(np.float32), axis=1)
            recording = np.clip(mono, -32768, 32767).astype(np.int16)
        elif recording.ndim == 2:
            recording = recording[:, 0]
        recording = np.asarray(recording, dtype=np.int16)
        wavfile.write(wav_path, SAMPLE_RATE, recording)
    except Exception as exc:
        raise RuntimeError(
            f"Audio capture failed for device {device_index}: {exc}\n"
            "Check the selected device, OS permissions, and loopback configuration. "
            "On Windows, try a WASAPI input/loopback-capable PortAudio build. "
            "On Linux, select a PulseAudio/PipeWire monitor device. "
            "On macOS, route system output through BlackHole and select its input."
        ) from exc

    LOG.info("Saved captured audio to temporary WAV (%s).", wav_path)
    if ambient_fallback:
        LOG.warning("This capture may contain ambient microphone audio, not system output.")

def transcribe_audio(
    whisper,
    wav_path: str,
    model_name: str
    ) -> tuple[str, str]:
    LOG.info("Loading Whisper model '%s' (first use may download model files)...", model_name)
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(wav_path, fp16=False)
    except Exception as exc:
        raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

    text = str(result.get("text", "")).strip()
    language = str(result.get("language", "unknown")).strip() or "unknown"
    return text, language


def translate_text(
    translator_class,
    text: str,
    target_lang: str) -> str:
    LOG.info("Translating detected speech to '%s'...", target_lang)
    try:
        translated = translator_class(source="auto", target=target_lang).translate(text)
    except Exception as exc:
        raise RuntimeError(
            "Translation failed. Check internet connectivity and the target language code. "
            f"Details: {exc}"
        ) from exc
    if not translated or not str(translated).strip():
        raise RuntimeError("Translation service returned empty text.")
    return str(translated).strip()
    

def synthesize_speech(
    gtts_class,
    text: str,
    target_lang: str,
    mp3_path: str
    ) -> None:
    LOG.info("Synthesizing translated voiceover...")
    try:
        tts = gtts_class(text=text, lang=target_lang)
        tts.save(mp3_path)
    except Exception as exc:
        raise RuntimeError(
            "Text-to-speech failed. Check internet connectivity and whether gTTS supports "
            f"the language code '{target_lang}'. Details: {exc}"
        ) from exc
    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
        raise RuntimeError("gTTS did not produce a usable MP3 file.")

        

def export_dub(
    pydub,
    wav_path: str,
    tts_path: str,
    output_path: str
    ) -> None:

    from pydub import AudioSegment

    try:
        original = AudioSegment.from_wav(wav_path)
        voice = AudioSegment.from_file(tts_path, format="mp3")
        silent_original = original - 120
        silent_original = AudioSegment.silent(
            duration=len(original),
            frame_rate=original.frame_rate,
        ).set_channels(original.channels).set_sample_width(original.sample_width)

        result = silent_original.overlay(voice, position=0)
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = destination.suffix.lower()
        if suffix == ".wav":
            result.export(str(destination), format="wav")
        elif suffix == ".mp3" or not suffix:
            result.export(str(destination), format="mp3", bitrate="192k")
        else:
            fmt = suffix.lstrip(".")
            result.export(str(destination), format=fmt)
    except Exception as exc:
        raise RuntimeError(
            f"Audio post-processing/export failed: {exc}. Ensure FFmpeg is installed "
            "and the output extension is supported by your FFmpeg build."
        ) from exc

    LOG.info("Exported dubbed audio: %s", destination)
    LOG.info(
        "Original duration: %.2fs; synthesized voice duration: %.2fs.",
        len(original) / 1000.0,
        len(voice) / 1000.0,
    )
    if len(voice) > len(original):
        LOG.warning(
            "The synthesized voice is longer than the original recording; "
            "the output will extend to fit the complete voiceover."
        )
        

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture system audio, transcribe it, translate it, and export a dubbed file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l", "--list-devices", action="store_true",
        help="List audio devices, show the automatic loopback candidate, and exit.",
    )
    parser.add_argument(
        "-d", "--duration", type=int, default=10,
        help="Capture duration in seconds.",
    )
    parser.add_argument(
        "-t", "--target-lang", default="bg",
        help="Target language code (e.g. bg, en, es, de, fr).",
    )
    parser.add_argument(
        "-o", "--output", default="output.mp3",
        help="Output file path (.mp3, .wav, or another FFmpeg-supported format).",
    )
    parser.add_argument(
        "-i", "--device", type=int, default=None,
        help="Explicit audio device index; skips automatic device selection.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, choices=("tiny", "base", "small", "medium", "large"),
        help="Whisper model size. Larger models are slower and use more memory.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )
    return parser
    


def main(
    argv: Optional[list[str]] = None
    ) -> int:

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.duration <= 0:
        parser.error("--duration must be a positive integer.")
    if not args.target_lang or not args.target_lang.strip():
        parser.error("--target-lang must not be empty.")

    try:
        deps = require_dependencies()
        sd = deps["sounddevice"]
        devices, auto_index = query_devices(sd)

        if args.list_devices:
            print_devices(sd, devices, auto_index)
            return 0

        check_ffmpeg()
        device_index, ambient_fallback = select_capture_device(
            sd, devices, auto_index, args.device
        )

        wav_fd, wav_path = tempfile.mkstemp(prefix="transdub_original_", suffix=".wav")
        os.close(wav_fd)
        mp3_fd, mp3_path = tempfile.mkstemp(prefix="transdub_tts_", suffix=".mp3")
        os.close(mp3_fd)

        try:
            capture_audio(
                sd, deps["numpy"], deps["wavfile"], device_index, args.duration,
                wav_path, ambient_fallback,
            )

            original_text, detected_lang = transcribe_audio(
                deps["whisper"], wav_path, args.model
            )
            if not original_text:
                LOG.info("[-] No speech detected in the audio sequence.")
                return 0

            LOG.info("Detected source language: %s", detected_lang)
            LOG.info("Recognized text: %s", original_text)

            translated_text = translate_text(
                deps["GoogleTranslator"], original_text, args.target_lang.strip().lower()
            )
            LOG.info("Translated text: %s", translated_text)

            synthesize_speech(
                deps["gTTS"], translated_text, args.target_lang.strip().lower(), mp3_path
            )
            export_dub(deps["pydub"], wav_path, mp3_path, args.output)
            return 0
        finally:
            for temporary_path in (wav_path, mp3_path):
                try:
                    if os.path.exists(temporary_path):
                        os.remove(temporary_path)
                except OSError as cleanup_exc:
                    LOG.warning(
                        "Could not remove temporary file %s: %s",
                        temporary_path, cleanup_exc,
                    )

    except KeyboardInterrupt:
        LOG.error("Interrupted by user.")
        return 130
    except Exception as exc:
        LOG.error("%s", exc)
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
