class SimpleTokenizer:
    """A simple one token per character tokenizer"""
    def __init__(self, text):
        # text -> Complete dataset in string form
        self.chars = sorted(list(set(text)))
        print(f"{len(self.chars)} different characters in total")
        
        # Look up tables words <-> token IDs
        self.encode_lut = { ch:i for i,ch in enumerate(self.chars) }
        self.decode_lut = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, text):
        return [self.encode_lut[ch] for ch in text]
    
    def decode(self, tokens):
        return ''.join([self.decode_lut[t] for t in tokens])