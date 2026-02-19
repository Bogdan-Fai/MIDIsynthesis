class Tokenizer:
    pitches_list = []
    vocab = {}
    vocab_size = 0
    
    @staticmethod
    def most_recent(number):
        if len(Tokenizer.vocab_size) == 0:
            print("Tokenize or load vocabulary first")
            return

        from collections import Counter
        
        cnt = Counter(Tokenizer.pitches_list)
        for token, n in cnt.most_common(number):
            print(n, token)

    @staticmethod
    def uniq_durations():
        if len(Tokenizer.vocab_size) == 0:
            print("Tokenize or load vocabulary first")
            return
        
        durs = set()
        for t in Tokenizer.pitches_list:
            if "_" in t:
                durs.add(t.split("_")[-1])
        print("Unique durations:", len(durs))
        print(sorted(durs))

    @staticmethod
    def chords_rate():
        if len(Tokenizer.vocab_size) == 0:
            print("Tokenize or load vocabulary first")
            return

        vocab = sorted(set(Tokenizer.pitches_list))
        chords = [t for t in vocab if "." in t]
        notes  = [t for t in vocab if "." not in t]

        print("Vocab:", len(vocab))
        print("Chords:", len(chords))
        print("Notes:", len(notes))
        print(float(len(chords)/len(vocab)) * 100, "%")

    @staticmethod
    def tokenize_all(midis_path):
        print("Tokenizing...")
        import os
        midis_list = os.listdir(midis_path)
        from music21 import converter, note, chord

        for midi_file in midis_list:
            if midi_file.endswith(".mid"):
                path = os.path.join(midis_path, midi_file)
                print(path)
                midi = converter.parse(path)

                for element in midi.flatten().notes:
                    if isinstance(element, note.Note):
                        Tokenizer.pitches_list.append(f"{element.pitch}_{element.quarterLength}")
                    elif isinstance(element, chord.Chord):
                        pitches = ""
                        for pitch in element.pitches:
                            pitches += str(pitch) + "_"
                        Tokenizer.pitches_list.append(pitches[:-1])
        
        vocab = {t: i for i, t in enumerate(sorted(set(Tokenizer.pitches_list)))}
        vocab_size = len(vocab)
        print("Tokenizing is ready. Vocabulary size: ", vocab_size)
        with open(os.path.join(midis_path, "vocab.txt"), "w") as f:
            for t, i in vocab.items():
                f.write(f"{t} {i}\n")

    @staticmethod
    def load_vocab(vocabulary_path):
        import os.path
        if os.path.exists(vocabulary_path):
            with open(vocabulary_path, "r") as f:
                for line in f:
                    t, i = line.split()
                    Tokenizer.vocab[t] = int(i)
            Tokenizer.vocab_size = len(Tokenizer.vocab)
            print("Vocabulary loaded. Size = ", Tokenizer.vocab_size)