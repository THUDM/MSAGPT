from typing import Sequence, Tuple, List, Union
import itertools

class ResidueLevelTokenizer:
    """
    Tokenizer for Protein Residue Level Tokenization.
    """

    def __init__(self, **kwargs):
        super(ResidueLevelTokenizer, self).__init__()
        self.pad_tok = ['[pad]']
        self.all_toks = self.pad_tok
        self._tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
        self.all_toks.extend(self._tokens)
        self._special_tokens = ['MASK', 'gMASK', 'sMASK', 'eod', 'sop', 'eop', '</s>', '<M>']    
        self.set_special_tokens(self._special_tokens)
        self.special_tokens['eos']=self.special_tokens['</s>']
        self.special_tokens['tMASK']=self.special_tokens['MASK']
        
        self.all_toks.extend(self._special_tokens) 
        self._vocab = {t: i for i, t in enumerate(self.all_toks)}
        self.command_token = {'[tMASK]': 'tMASK', '[MASK]':'MASK', '[gMASK]': 'gMASK', '[sMASK]':'sMASK'}
        # print('Building vocab.: {}'.format(self._vocab))
        # print('Special_tokens: {}'.format(self.special_tokens))
        # print('All tokens: {}'.format(self.all_toks))

    def pad_id(self):
        return self._vocab['[pad]']
    
    def set_special_tokens(self, special_tokens):
        """Add a list of additional tokens to the encoder.
        The additional tokens are indexed starting from the last index of the
        current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, len(self.all_toks) + i) for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}
        
        
    def __len__(self):
        return len(self._vocab)


    def EncodeAsIds(self, text, process_fn=None):
        """convert sequence to idx"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
            processed_text = str(processed_text)
        tokens = [self.TokenToId(c) for c in processed_text]
        return tokens
    
    def IdToToken(self, idx):
        if idx == 0:
            return '[pad]'
        elif idx in self.special_tokens_decoder:
            return f"[{self.special_tokens_decoder[idx]}]"
        else:
            try:
                tok = self.all_toks[idx]
            except:
                tok = '*'
            return tok
    def TokenToId(self, token):
        if token == '[pad]':
            return 0
        elif token in self.special_tokens:
            return self.special_tokens[token]
        else:
            return self._vocab[token]
    
    def DecodeIds(self, Ids):
        return ''.join([self.IdToToken(tok) for tok in Ids])
    
    def _tokenize(self, text) -> str:
        return text.split()
    
    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self._tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.all_toks
                        else [token]
                        for token in tokenized_text
                    )
                )
            )
        no_split_token = self.all_toks
        tokenized_text = split_on_tokens(no_split_token, text)
        return self.convert_tokens_to_ids(tokenized_text)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        # print_rank_0(tokens)
        # print_rank_0(self.vocab)
        for token in tokens:
            ids.append(self.TokenToId(token))
        return ids


class proteinglm_tokenizer:
    """
    Protein Tokenizer based on Residue level tokenizer
    """

    def __init__(self):
        name = 'ProteinTokenizer'
        self.tokenizer = ResidueLevelTokenizer()
        self.special_tokens = self.tokenizer.special_tokens


    def IdToToken(self, idx):
        return self.tokenizer.IdToToken(idx)

    def TokenToId(self, token):
        return self.tokenizer.TokenToId(token)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def decode(self, token_ids):
        return self.tokenizer.DecodeIds([token_ids])

    @property
    def eod(self):
        return self.tokenizer.get_special_token('eos')

    def detokenize(self, Ids, type_token=False):
        new_tokens = self.tokenizer.DecodeIds(Ids)
        return new_tokens

    def tokenize(self, text):
        ids = self.tokenizer.tokenize(text)
        return ids

    @property
    def vocab(self):
        return self.tokenizer._vocab

    @property
    def inv_vocab(self):
        return {v:k for k, v in self.tokenizer._vocab.items()}

    @property
    def get_pad_id(self):
        return self.tokenizer.pad_id
    
    
    def get_command(self, token):
        tok = token
        if token in self.tokenizer.command_token:
            tok = self.tokenizer.command_token[token]
        return self.tokenizer.special_tokens[tok]
