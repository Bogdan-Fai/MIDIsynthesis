from ctypes import sizeof
from music21 import converter, note, chord

def mid2txt(midi_file, txt_file=""):
    """
    Converts .mid file to .txt with supported format.
    """

    if txt_file == "":
        txt_file = midi_file.replace(".mid", ".txt")
    elif not txt_file.endswith(".txt"):
        print(str(txt_file).split("/")[-1], "is not a .txt file!")
        return
    if not midi_file.endswith(".mid"):
        print(str(midi_file).split("/")[-1], "is not a .mid file!")
        return

    midi = converter.parse(midi_file)
    text = ""

    for element in midi.flatten().notes:
        if isinstance(element, note.Note):
            text += str(element.pitch) + " " + str(element.quarterLength) + "\n"
        elif isinstance(element, chord.Chord):
            pitches = ""
            for pitch in element.pitches:
                pitches += str(pitch) + " "
            text += pitches + "\n"

    with open (txt_file, "w") as f:
        f.write(text)

    return text

def txt2mid(txt_file: str = "", midi_file: str = "", text:str=[]):
    """
    Converts tokens to MIDI.

    Supported token format (one token per line):
      - Note:  C4_0.5
      - Chord: C4.E4.G4

    You can pass either:
      - txt_file="path/to/file.txt"
      - OR text=[ "C4_0.5", "C4.E4.G4", ... ]
    """

    # determine output midi path
    if not midi_file:
        if txt_file:
            midi_file = txt_file.replace(".txt", ".mid")
        else:
            midi_file = "output.mid"

    if not midi_file.endswith(".mid"):
        raise ValueError(f"{midi_file} is not a .mid file")

    # read tokens
    tokens = []
    if txt_file:
        if not txt_file.endswith(".txt"):
            raise ValueError(f"{txt_file} is not a .txt file")
        with open(txt_file, "r", encoding="utf-8") as f:
            tokens = [line.strip() for line in f if line.strip()]
    else:
        if not text:
            raise ValueError("No sources! Provide txt_file or text list.")
        tokens = [str(t).strip() for t in text if str(t).strip()]

    from music21 import stream, note, chord, duration

    s = stream.Stream()

    for tok in tokens:
        if  "." in tok:  # chord like C4.E4.G4
            n = note.Note(tok.split(" ")[0])
            n.quarterLength = float(tok.split(" ")[1])
            s.append(n)

        else:  # single note like C4
            pitches = tok.split(" ")
            c = chord.Chord(pitches)
            s.append(c)

    s.write("midi", fp=midi_file)