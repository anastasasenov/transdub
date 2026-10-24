# AI Translator-Dubber: Cross-Platform Offline Dubbing and Voice Cloning Tool
![AI Assisted](https://shields.io)

This is a cross-platform command-line tool that captures system audio, transcribes it using OpenAI Whisper, translates it into a target language, synthesizes a voiceover, and exports a dubbed audio file.

## Prerequisites

* Python 3.8+
* FFmpeg: Must be installed and available on your system's PATH. Required by pydub and whisper.

### Python Dependencies

    pip install "sounddevice>=0.4.6" "numpy>=1.22" "scipy>=1.8" \
        "openai-whisper>=20230314" "deep-translator>=1.11.4" \
        "gTTS>=2.3.2" "pydub>=0.25.1"

## Usage

    $ python3 transdub.py [options]
   
    Command-Line Options:
    -l, --list-devices: List all available audio devices, show the automatically selected loopback candidate, and exit.
    -d, --duration DURATION: Capture duration in seconds (default: 10).
    -t, --target-lang LANG: Target language code for translation and text-to-speech, e.g., bg, en, es, de, fr (default: bg).
    -o, --output PATH: Output file path. Supports .mp3, .wav, or other FFmpeg-compatible formats (default: output.mp3).
    -i, --device INDEX: Explicit audio device index; skips automatic device selection.
    --model MODEL: Whisper model size (tiny, base, small, medium, large). Larger models require more memory and are slower (default: base).
    -v, --verbose: Enable debug logging and full tracebacks on error.
   
### Examples

List available audio devices to find your system loopback/monitor:

    $ python transdub.py --list-devices
   
Capture 15 seconds of system audio, translate to Spanish (es), and save as dubbed.mp3 using the small Whisper model:

    $ python transdub.py --duration 15 --target-lang es --output dubbed.mp3 --model small

## License
This project is open-source and distributed under the MIT License.

