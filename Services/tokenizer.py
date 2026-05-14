import os.path
import json

class Tokenizer:
    pitches_list = []
    vocab = {}
    vocab_loaded = False
    vocab_size = 0
    
    #
    # Токенизация всех MIDI файлов в папке
    #
    def build_vocab(self,midis_path, trace=False):
        self.pitches_list = []
        self.vocab = {}
        self.vocab_size = 0

        print("Tokenizing...")
        midis_list = sorted(os.listdir(midis_path))
        from music21 import converter, note, chord

        for midi_file in midis_list:
            if midi_file.endswith(".mid"):
                path = os.path.join(midis_path, midi_file)
                if trace: 
                    print(path)
                midi = converter.parse(path)

                for element in midi.flatten().notes:
                    if isinstance(element, note.Note):
                        self.pitches_list.append(f"{element.pitch}_{element.quarterLength}")
                    elif isinstance(element, chord.Chord):
                        pitches = ""
                        for pitch in element.pitches:
                            pitches += str(pitch) + "_"
                        pitches += str(element.quarterLength)
                        self.pitches_list.append(pitches)
                    elif isinstance(element, note.Rest):
                        self.pitches_list.append(f"REST_{element.quarterLength}")
        
        self.vocab = {t: i+3 for i, t in enumerate(sorted(set(Tokenizer.pitches_list)))}
        self.vocab["START"] = 0
        self.vocab["END"] = 1
        self.vocab["PAD"] = 2
        self.vocab_size = len(self.vocab)
        json.dump(self.vocab, open("Data/vocab.json", "w"), ensure_ascii=False, indent=4)
        # with open("Data/vocab.txt", "w") as f:
        #     for t, i in Tokenizer.vocab.items():
        #         f.write(f"{t} {i}\n")
        self.vocab_loaded = True

        print("Tokenizing is ready. Vocabulary size: ", self.vocab_size)

    def tokenize_file(self, midi_file, output_type="ids"):
        if not self.vocab_loaded:
            print("Vocabulary not loaded. Load it first with load_vocab(path) or tokenize all MIDI files using build_vocab(path).")
            return None
        if output_type not in ["tokens", "ids"]:
            print("Invalid output type! Use 'tokens' or 'ids'.")
            return None
        from music21 import converter, note, chord
        
        tokens = []
        tokens.append("START")
        midi = converter.parse(midi_file)

        for element in midi.flatten().notes:
            if isinstance(element, note.Note):
                tokens.append(f"{element.pitch}_{element.quarterLength}")
            elif isinstance(element, chord.Chord):
                pitches = ""
                for pitch in element.pitches:
                    pitches += str(pitch) + "_"
                pitches += str(element.quarterLength)
                tokens.append(pitches)
        
        tokens.append("END")
        if output_type == "tokens":
            return tokens
        elif output_type == "ids":
            return self._encode(tokens)

    def _encode(self, tokens):
        if not self.vocab_loaded:
            print("Vocabulary not loaded. Load it first with load_vocab(path) or tokenize all MIDI files using build_vocab(path).")
            return []
        token_ids = []
        for t in tokens:
            if t in self.vocab:
                token_ids.append(self._token_to_id(t))
            else:
                print(f"Token '{t}' not found in vocabulary!")
                token_ids.append(None)
        return token_ids

    def _decode(self, token_ids):
        if not self.vocab_loaded:
            print("Vocabulary not loaded. Load it first with load_vocab(path) or tokenize all MIDI files using build_vocab(path).")
            return []
        tokens = []
        for id in token_ids:
            token = self._id_to_token(id)
            if token is not None:
                tokens.append(token)
            else:
                print(f"ID '{id}' not found in vocabulary!")
                tokens.append(None)
        return tokens

    #
    # Загрузка словаря из файла .txt
    #
    def load_vocab(self, vocab_path):
        if os.path.exists(vocab_path):
            self.vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
            self.vocab_size = len(self.vocab)
            self.vocab_loaded = True

            print("Vocabulary loaded. Size = ", self.vocab_size)
        else:
            print("Path isn't exists!")

    def _token_to_id(self, token):
        if token in self.vocab:
            return self.vocab[token]
        else:
            print("Token not found in vocabulary!")
            return None
        
    def _id_to_token(self, id):
        for t, i in self.vocab.items():
            if i == id:
                return t
        print("ID not found in vocabulary!")
        return None


    #
    # Вывод самых часто встречающихся токенов
    #
    def _most_recent(self, number):
        if self.pitches_list:
            from collections import Counter
            
            cnt = Counter(token for token,i in self.vocab)
            for token, n in cnt.most_common(number):
                print(n, token)
        else:
            print("Tokenize first!")

    def _uniq_durations(self):
        if len(self.vocab_size) == 0:
            print("Tokenize or load Tokenizer first")
            return
        
        durs = set()
        for t in self.pitches_list:
            if "_" in t:
                durs.add(t.split("_")[-1])
        print("Unique durations:", len(durs))
        print(sorted(durs))

    def _chords_rate(self):
        if len(self.vocab_size) == 0:
            print("Tokenize or load Tokenizer first")
            return

        vocab = sorted(set(self.pitches_list))
        chords = [t for t in vocab if "." in t]
        notes  = [t for t in vocab if "." not in t]

        print("Vocab:", len(vocab))
        print("Chords:", len(chords))
        print("Notes:", len(notes))
        print(float(len(chords)/len(vocab)) * 100, "%")