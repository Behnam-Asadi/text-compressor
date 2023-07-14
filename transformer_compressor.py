import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TransformerCompressor:

    def __init__(self, model, vocab_size: int, pad_token: int, eos_token: int):
        self.model = model
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.model.eval()
        self.eos_token = eos_token

    def valid_encodings(self, shifted_inputs, encoded_msgs, sorted_tokens):
        # At each timestep, use the encoded message to select the tokens at the specified
        # index of the list of sorted tokens to reconstruct the original message.
        # Compare against the original message to ensure they are identical.
        batch_size, token_len, vocab_size = sorted_tokens.size()
        msg_len = token_len - 1

        # Flatten the tensor of sorted tokens to make indexing easier
        # and add offsets to the encoded message to account for this flattening
        sorted_tokens_flat = sorted_tokens.view(batch_size, -1)
        encoded_msgs_offset = encoded_msgs + torch.arange(0, vocab_size * msg_len, vocab_size).to(device)
        decoded_msgs_cand = torch.gather(sorted_tokens_flat, 1, encoded_msgs_offset)
        return torch.all(decoded_msgs_cand == shifted_inputs[:, :-1])

    def trans_encode(self, tokenized_msgs, attentions):
        """
    Parameters
      tokenized_msgs: shape (batch_size, msg_len)
      attentions: shape (batch_size, msg_len)

    """
        # Encode
        tokenized_msgs = tokenized_msgs.to(device)
        attentions = attentions.to(device)
        model = self.model
        with torch.no_grad():
            # In theory, I should be able to avoid the loop because the transformer
            # automatically masks the input. But in practice, this causes the logit
            # outputs to differ slightly between the encoder and decoder
            batch_size, msgs_len = tokenized_msgs.size()
            logits_arr = torch.zeros(batch_size, msgs_len, self.vocab_size).to(device)
            for i in range(msgs_len):
                msgs_slice = tokenized_msgs[:, :i + 1]
                attentions_slice = attentions[:, :i + 1]
                logits = model(msgs_slice, attention_mask=attentions_slice).logits
                logits_arr[:, i] = logits[:, i]

        # Sort the indices of the logits in descending order of logit value.
        # This means that the model's top predicted token is the first
        # element in the sorted list, the second highest predicted token is the
        # second element, and so on.
        #
        # Once we have this list of tokens ordered by their probability
        # we can find the ground-truth token in this list, and save its index
        # as the encoding of the token.
        shifted_inputs = torch.roll(tokenized_msgs, -1)  # Shift inputs to line up with output
        _, sorted_tokens = torch.sort(logits_arr, dim=2, descending=True, stable=True)
        shifted_inputs_reshaped = shifted_inputs.view(batch_size, msgs_len, 1)
        encoded_msgs = (sorted_tokens == shifted_inputs_reshaped).nonzero()[:, 2].reshape(batch_size, -1).to(device)
        encoded_msgs = encoded_msgs[:, :-1]  # Discard the last index because it overflows the original message
        assert self.valid_encodings(shifted_inputs, encoded_msgs, sorted_tokens)

        # We need to include the first token as part of the encoded message so that we
        # can bootstrap generation
        encoded_msgs = torch.cat((tokenized_msgs[:, :1], encoded_msgs), dim=1)

        return encoded_msgs, logits_arr  # Logits for debugging

    def trans_decode(self, encoded_msgs):

        encoded_msgs = encoded_msgs.to(device)
        model = self.model

        with torch.no_grad():
            # The first value in the encoded message
            # is the first token of the original message
            first_tokens = encoded_msgs[:, :1]
            encoded_msgs = encoded_msgs[:, 1:]

            batch_size, msg_len = encoded_msgs.size()
            logits_arr = torch.zeros(batch_size, msg_len, self.vocab_size).to(device)  # For debugging
            decoded_msgs = first_tokens
            for i in range(msg_len):
                logits = model(decoded_msgs).logits
                logits_arr[:, i] = logits[:, i]  # For debugging
                _, indices = torch.sort(logits[:, i, :], dim=1, descending=True, stable=True)
                decoded_tokens = torch.gather(indices, 1, encoded_msgs[:, i:i + 1])
                decoded_msgs = torch.cat((decoded_msgs, decoded_tokens), dim=1)
        return decoded_msgs, logits_arr  # Logits for debugging

    def verify_msgs(self, decoded_msgs, original_msgs, attentions):
        attentions_bool_mask = attentions.type(torch.BoolTensor).to(device)
        pad_token_mask = torch.ones(decoded_msgs.size(), dtype=int).to(device) * self.pad_token
        # We do this masking because the decompressor will spit out garbage output
        # after the end of a message but we don't care about this because we can identify
        # end-of-message by looking for the first padding token.
        decoded_msgs_cleaned = torch.where(attentions_bool_mask, decoded_msgs, pad_token_mask)
        return torch.all(decoded_msgs_cleaned == original_msgs)
