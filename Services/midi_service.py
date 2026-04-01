def play_midi(path):
    import pygame

    pygame.init()
    pygame.mixer.init()

    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

def play_midi_from_stream(s):
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
        s.write("midi", fp=tmp.name)
        play_midi(tmp.name)
        os.unlink(tmp.name)

def save_midi_from_stream(s, path="output.mid"):
    s.write("midi", fp=path)

def see_notes(path):
    from music21 import converter, note, chord

    midi = converter.parse(path)

    for element in midi.flatten().notes:
        if isinstance(element, note.Note):
            print("Note:", element.pitch, "Duration:", element.quarterLength)
        elif isinstance(element, chord.Chord):
            print("Chord:", element.pitches)

def play_sequence(sequence, path = "Data/test.mid"):
    from music21 import stream, note
    s = stream.Stream()

    for pitch in sequence:
        n = note.Note(pitch)
        n.quarterLength = 2.0
        s.append(n)

    s.write("midi", fp=path)
    play_midi(path)