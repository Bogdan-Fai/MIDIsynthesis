from Services.midi_service import play_midi, see_notes, play_sequence

def main():
    melody = ["C4", "E4", "G4", "C5", "G4", "E4", "C4"]
    play_sequence(melody)

if __name__ == "__main__":
    main()