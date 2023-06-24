#  Copyright (c) 2022. Code by DJohoe28.
from Optimizer import *

import argparse
import time

from datetime import timedelta

parser = argparse.ArgumentParser(
    prog="ACR Project",
    description="Automatic Content Recognition of songs in Hebrew.",
    epilog="By Jonathan Eddie Amir for Tel Hai College."
)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-p', '--path', default=OPTIONS["RecordingPath"])
parser.add_argument('-e', '--extension', default=OPTIONS["RecordingExtension"])
parser.add_argument('-sr', '--sample-rate', default=OPTIONS["SampleRate"], type=Integer)
parser.add_argument('-c', '--channels', default=OPTIONS["Channels"], type=Integer)
parser.add_argument('-n', '--num-fft', default=OPTIONS["NFFT"], type=Integer)
parser.add_argument('-l', '--hop-length', default=OPTIONS["HopLength"], type=Integer)
parser.add_argument('-w', '--window-length', default=OPTIONS["WindowLength"], type=Integer)
parser.add_argument('-th', '--threshold', default=OPTIONS["Threshold"], type=Integer)
parser.add_argument('-d', '--distance', default=OPTIONS["Distance"], type=Integer)
# TODO: Add new OPTIONS members.
args = parser.parse_args()

if not OPTIONS["COLAB"]:
    sd.default.samplerate = OPTIONS["SampleRate"]  # Currently unused, as we know each file's sample rate
    sd.default.channels = OPTIONS["Channels"]  # Set sound-device's default to Stereo


# Utilities
def example_load() -> (plt.Figure, plt.Axes):
    """
    Loads the 'choice' and 'nutcracker' example file from the librosa example library as Track instances.

    :return: Plotted figure & axes
    :rtype: (plt.Figure, plt.Axes)
    """
    _track1 = Track(librosa.ex('choice', hq=True))
    _track2 = Track(librosa.example('nutcracker'))
    _fig, _axs = plt.subplots(2, 2)
    _track1.plot_stft_and_constellation(axs=_axs[0], fig=_fig)
    _track2.plot_stft_and_constellation(axs=_axs[1], fig=_fig)
    return _fig, _axs


# Main Program
def main() -> Return:
    """
    The Main() function. Provides a user interface for the program.

    :return: Did the function complete successfully?
    :rtype: Return
    """
    # Variables
    print("Hello! Welcome to the ACR Project!")
    running = True
    database: TrackList = TrackList()
    database.hashes = load_from_pickle("hash.npy")  # TODO: Loads hashes to memory - super important! Check if works!!

    # Commands
    def load() -> Return:
        """Initialize a demo Track list."""
        nonlocal database
        _input = input("Would you like to import a song directory into the database? [y]es / [n]o: ")
        if _input.lower() in ["true", "t", "yes", "y"]:
            OPTIONS["DatabasePath"] = input("Please specify the path to the directory of songs: ")
            cache_database_by_track()
            print("Database loaded successfully!")
        else:
            _input = input("Please specify the path to the cache file you want added to the database: ")
            try:
                _temp = np.load(_input)
                for k in _temp:
                    if k not in database.hashes:
                        database.hashes[k] = {}
                    database.hashes[k] += _temp[k]
                print("Cache loaded successfully!")
            except FileNotFoundError:
                print("File not found, please try again later.")
                return Return.FILES_ERROR
        return Return.SUCCESS

    def terminate() -> Return:
        """Terminate the program."""
        nonlocal running
        stop()
        running = False
        return Return.SUCCESS

    def play() -> Return:
        """Play a Track by ID from user."""
        print("Available Tracks:")
        for _i in range(len(database)):
            print(f"{_i}: {database[str(_i)].title}")
        _track_id = input("Please enter Track ID:")
        try:
            _track_id = int(_track_id)
            if _track_id == "":
                print("Cancelling...")
                return Return.CANCEL
            _track: Track = database[str(_track_id)]  # TODO: Search for names
            _track.play()
            return Return.SUCCESS
        except ValueError:
            print("Not a valid Track ID number. Please try again.")
            return Return.VALUE_ERROR
        except IndexError:
            print("Track ID out of range. Please try again.")
            return Return.INDEX_ERROR
        except sd.PortAudioError:
            print("Error playing Track. Please try again.")
            return Return.AUDIO_ERROR
        pass

    def stop() -> Return:
        """Stop playing all currently playing audios on sound-device."""
        if OPTIONS["COLAB"]:
            return Return.COLAB_ERROR
        sd.stop()
        return Return.SUCCESS

    def record(wait: bool = True, save: bool = True, add: bool = False, ext: AnyStr = OPTIONS["FileType"]) -> Return:
        """Record from sound-device."""
        if OPTIONS["COLAB"]:
            print("This functionality is unavailable for Google Colab. Please use a sample file, instead.")
            return Return.COLAB_ERROR
        try:
            _duration = float(
                input(f"Please enter recording Duration (seconds, Default={OPTIONS['DefaultDuration']}):") or OPTIONS[
                    "RecordingDuration"])
        except ValueError:
            print("Duration invalid. Please try again.")
            return Return.VALUE_ERROR
        if OPTIONS["SampleRate"] is None:
            try:
                _sr = int(input(f"Please enter Sample Rate (Hz, Default={OPTIONS['DefaultSampleRate']}):") or OPTIONS[
                    "SampleRate"])
            except ValueError:
                print("Sample Rate invalid. Please try again.")
                return Return.VALUE_ERROR
        else:
            _sr = OPTIONS["SampleRate"]
        _recording = sd.rec(int(_duration * _sr))
        if wait:
            sd.wait()  # Wait for recording to finish.
        _track = Track(f'{OPTIONS["RecordingPath"]}/Recording - {timestamp()}.{OPTIONS["RecordingExtension"]}')
        _track._sr = _sr
        _track._y = _recording
        if save:
            if not os.path.exists(OPTIONS["RecordingPath"]):
                # Create "Recordings" folder if one does not exist. (SoundFile can't mkdir)
                os.mkdir(OPTIONS["RecordingPath"], mode=OPTIONS["MakeDirMode"])
            sf.write(f"{_track.title}.{ext}", _track.y, _track.sr)  # Save recording to file.
        # TODO: Soundfile takes a while to save the recording - is it possible to block Python in the meantime?
        if add:
            database.append(_track)  # TODO: Make sure this works.
        return Return.SUCCESS

    def append() -> Return:
        """Append a new Track to the database using soundfile."""
        _path = input("Please enter the filepath of the Track you would like to add (.wav, 44100Hz SR): ")
        if sf.check_format(_path):
            _track = Track(_path)
            database.append(_track)
            return Return.SUCCESS
        return Return.FILES_ERROR

    def match() -> Return:
        """Search for a Track using a Sample soundfile."""
        _path = input("Please enter the filepath of the Sample you would like to search with (.wav, 44100Hz SR): ")
        if sf.check_format(_path):
            _track = Track(_path)
            _hashes = _track.hashes
            _title = database.match_from_sample(sample=_hashes)
            print(f"Match Found! {_title}")
            return Return.SUCCESS
        return Return.INDEX_ERROR  # No match found.

    def optimize() -> Return:
        """Optimizes various window parameters."""
        print("Please wait while optimizing...")
        best_params, best_score = optimize_parameters()
        print("Training Results:", best_params, best_score)
        return Return.SUCCESS

    # Command List
    _commands = dict()
    for _command in [load, terminate, play, stop, record, append, match, optimize]:  # TODO: Add all commands here!
        _commands[_command.__name__] = _command

    # Main Loop
    while running:
        print("\nAvailable Commands:")
        print(list(_commands.keys()))
        _command_name = input("Please enter a command:").lower()  # Case-insensitive.
        if _command_name not in _commands.keys():
            print("Command not found. Please try again.")
            continue
        _status_code = _commands[_command_name]()  # TODO: Handle status code
    print("Goodbye!")
    return Return.SUCCESS


if __name__ == "__main__":
    pass  # main()

# TODO:
#  COMPLETE: Change Sample Rate -> Higher SR = shorter higher pitch
#  COMPLETE: Experiment with cross-referencing librosa, soundfile, python-sounddevice -> All are based on NumPy.
#  COMPLETE: Basic Menu -> Currently good-faith programmer-only.
#  COMPLETE: Playback -> Not tested with relative/absolute path.
#  COMPLETE: Recording (python-sounddevice = NumPy / pyaudio = byte) -> Used python-sounddevice, as it's based on NumPy.
#  COMPLETE: Change STFT window size
#  COMPLETE: Initial database -> Ripped Home CDs, will rip friends' CDs in the future.
#  COMPLETE: Options struct
#  COMPLETE: Show patch as window (8x8, 3x3, ...). Use np.max. Control target area / k parameters. (5 sec, 1024 bins)
#  COMPLETE: Window - 140ms ~ 500ms, ~10 in range, histogram points in window, threshold,
#  IN PROGRESS: Play with peak_find parameters to affect Constellation.
#  NOTE: Mode5, MDA256, CRC checksum
#  NOTE: matlab find, SIMD, parameter sweep, scipy research
# NOTE: This is the code I was working on to properly set up the database.
if not OPTIONS["DEBUG"]:
    main()
else:
    print(f"Starting! {timestamp()}")
    pickle_path: AnyStr = "C:/Users/DJoho/PycharmProjects/ACR_Project/Cache/Pickles/אביגייל רוז - הפרעות - לא טוב לי.pkl"
    track: Track = Track("C:/Users/DJoho/Downloads/_Database ACR Shazam/אביגייל רוז - הפרעות - לא טוב לי.wav")
    start: Float = time.time()
    cached: bool = OPTIONS["CACHED"]
    titles: List[AnyStr] = get_titles(cached) if cached else cache_database_by_track(OPTIONS["DatabasePath"], True)
    titles = get_titles()
    hashes = {}
    for filename in os.listdir(OPTIONS["DatabasePath"]):
        if not filename.endswith(OPTIONS["RecordingExtension"]):
            continue
        print(f"Hashing {filename}...")
        track = Track(os.path.join(f'{OPTIONS["DatabasePath"]}/{filename}'))
        _hashes = track.hashes  # hashes_from_cached_constellation(title)
        for key in _hashes:
            if key not in hashes:
                hashes[key] = {}
            for _title in _hashes[key]:
                hashes[key][_title] = _hashes[key][_title]
    save_as_pickle("Hashes.pkl", hashes)
    # tracklist: TrackList = tracklist_from_directory()
    runtime: Float = time.time() - start
    print(f"Done! {timedelta(seconds=runtime)} = {len(titles)} songs loaded!")
