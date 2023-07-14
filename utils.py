from transformer_compressor import TransformerCompressor
from transformers import PreTrainedTokenizerFast
import torch
import numpy as np
import math
import struct
import varints
from tqdm import tqdm


def get_residuals(msg: str, detokenized_message: str) -> dict:
    """
    Given a msg, and the detokenized message, 
    return a dictionary of residuals necessary for converting the 
    detokenized message into the message.

    Params:
        msg: str
            the message
        detokenized_message: str
            detokenized_message = tokenizer.decode(tokenizer.encode(msg))

    returns: 
        residuals: dict[int:char]
            each key is an integer representing an index in which
            to insert a character and each value is the character 
            to insert
    """
    residuals = {}
    # Used to track that the residuals successfully transform the 
    # detokenized message back into the original message
    fixed_decoded = []
    offset = 0
    for j in range(len(msg)):
        if msg[j] != detokenized_message[j - offset]:
            offset += 1
            fixed_decoded.append(msg[j])
            residuals[j] = msg[j]
        else:
            fixed_decoded.append(detokenized_message[j - offset])
    fixed_decoded = ''.join(fixed_decoded)
    assert fixed_decoded == msg
    return residuals


def apply_residuals(detokenized_message: str, residuals: dict) -> str:
    """
    Given a detokenized message and a dictionary of residuals
    apply the residuals to the detokenized message to obtain the original
    message.

    Parameters
        detokenized_message: str
            detokenized_message = tokenizer.decode(tokenizer.encode(msg))
        residuals: dict[int:char]
            each key is an integer representing an index in which
            to insert a character and each value is the character 
            to insert
    Returns
        msg: str
            - the original message
    
    """
    reconstituted = []
    offset = 0
    for i in range(len(detokenized_message) + len(residuals)):
        if i in residuals:
            reconstituted.append(residuals[i])
            offset += 1
        else:
            reconstituted.append(detokenized_message[i - offset])
    return ''.join(reconstituted)


class MsgPacker:

    @classmethod
    def binary_pack(cls, binarized_msg: bytes, residuals: dict) -> bytes:
        """
        Given a binarized message, and the message residuals
        pack the message and its residuals into a single binary string

        Parameters
            binarized_msg: bytes
                The message, which has already been binarized
        
            residuals: dict
                the information necessary to reconstruct the original message after decoding
                and detokenizing

        Returns
            packed: bytes
                the encoded message and residuals packed into a binary string
        """

        # Turn the encoded doc into a varint array encoded as bytes
        residual_keys_encoded = varints.encode(list(residuals.keys()))
        residual_vals_encoded = bytes(''.join(residuals.values()), 'utf-8')
        msg_num_bytes = len(binarized_msg)
        residual_keys_num_bytes = len(residual_keys_encoded)
        residual_vals_num_bytes = len(residual_vals_encoded)

        # Field one: Length of field four
        # Field two: Length of field five
        # Field three: Length of field six
        # Field four: Varint array of the keys of the residual dictionary
        #   Once decoded, each element of the array is a key of the dictionary of residuals
        #   The first element of the key array is matched with the first element of the value
        #   array to create the key-value pairs for the residual dictionary.
        # Field five: Character array of of the values of the residual dictionary
        #   Each characater is a value of the dictionary of residuals.
        #   Combined with information from field two to create residual key-val pairs.
        #   The characters location in the array is used to match it to the correct
        #   index in field two.
        # Field six: Varint array of document encoding
        format = 'III{0}s{1}s{2}s'.format(residual_keys_num_bytes, residual_vals_num_bytes, msg_num_bytes)
        packed = struct.pack(format, residual_keys_num_bytes, residual_vals_num_bytes, msg_num_bytes,
                             residual_keys_encoded, residual_vals_encoded, binarized_msg)
        return packed

    @classmethod
    def binary_unpack(cls, packed):
        """
        Given a binary string representing an encoded message and its residuals
        unpack it and return the binary-encoding of the message a dictionary of 
        residuals

        Parameters
            packed: bytes
                the encoded message and residuals packed into a binary string
        
        Returns
            binarized_msg: bytes
                The binarized message, which will need to be debinarized
            residuals: dict
                the information necessary to reconstruct the original message after decoding
                and detokenizing
        """

        UNSIGNED_INT_SIZE = 4
        # Header composed of three 4-byte unsigned integers
        # Payload comprises rest of the packed data
        header = packed[:3 * UNSIGNED_INT_SIZE]
        payload = packed[3 * UNSIGNED_INT_SIZE:]

        residual_keys_num_bytes, residual_vals_num_bytes, msg_num_bytes = struct.unpack('III', header)
        format = '{0}s{1}s{2}s'.format(residual_keys_num_bytes, residual_vals_num_bytes, msg_num_bytes)

        residual_keys_encoded, residual_vals_encoded, binarized_msg = struct.unpack(format, payload)
        if residual_vals_num_bytes > 0:
            residual_keys_decoded = varints.decode(residual_keys_encoded)
            residual_vals_decoded = [char for char in residual_vals_encoded.decode('utf-8')]
            residuals = {k: v for k, v in zip(residual_keys_decoded, residual_vals_decoded)}
        else:
            residuals = {}
        return binarized_msg, residuals


class MsgBinaryEncoder:

    @classmethod
    def varint_binarize(cls, trimmed_encoded_msg: torch.Tensor):
        """
        Binarize the msg

        Parameters
            encoded_msg_trimmed: torch.Tensor
                The transformer-encoded message where each value in the tensor represents
                the index of the correct token within a list of sorted logits. Unpadded

        Returns
            varint_binarized_msg: bytes
        """
        return varints.encode(trimmed_encoded_msg.tolist())

    @classmethod
    def varint_debinarize(cls, msg_bytes: bytes):
        """
        De-binarize the msg

        Paramters
            msg_bytes: bytes
                The varint-encoded binary message

        Returns 
            encoded_msg_trimmed: torch.Tensor
                The transformer-encoded message where each value in the tensor represents
                the index of the correct token within a list of sorted logits. Unpadded
        """
        return torch.tensor(varints.decode(msg_bytes), dtype=int)

    @classmethod
    def trim_padding(cls, encoded_msg: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        """
        Parameters
            encoded_msg: torch.Tensor
                The transformer-encoded message where each value in the tensor represents
                the index of the correct token within a list of sorted logits. Right-padded
            attention:  torch.Tensor
                Attention for the encoded message. The message may contain padding at the end
                and we do not want to pack this beause it is irrelevant data. The attention
                tells us where the message ends and the padding begins
        Returns
            encoded_msg_trimmed: torch.Tensor
                Message trimmed to the length specified by the attention

        """
        msg_len = torch.sum(attention)  # Number of 1s in attention is equal to doc length
        return encoded_msg[:msg_len]


def batched_encode(compressor: TransformerCompressor, tokenizer: PreTrainedTokenizerFast, dataset, batch_size,
                   slice_len):
    data_by_buckets = {}
    encodings_by_buckets = {}

    def tokenize_and_categorize(elem):
        res = tokenizer(elem['article'], return_tensors='np')

        input_ids = res['input_ids'].reshape(-1)
        attention = res['attention_mask'].reshape(-1)
        detokenized = tokenizer.decode(input_ids)

        pad_length = slice_len - (len(input_ids) % slice_len)
        padding = np.ones(pad_length, dtype=int)
        padded_input = np.concatenate((input_ids, padding * compressor.eos_token), axis=0)
        padded_attention = np.concatenate((attention, padding * 0))

        return {'length_bucket': len(padded_input), 'input_ids': padded_input, 'attention_mask': padded_attention,
                'detokenized': detokenized}

    tokenized_subset = dataset.map(
        tokenize_and_categorize, load_from_cache_file=False
    )

    for bucket in sorted(set(tokenized_subset['length_bucket'])):
        data_by_buckets[bucket] = tokenized_subset.filter(lambda elem: elem['length_bucket'] == bucket)

    print(f'Buckets: {list(data_by_buckets.keys())}')
    for bucket in data_by_buckets.keys():
        same_len_subset = data_by_buckets[bucket]
        same_len_subset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        num_rows = same_len_subset.num_rows
        trans_encodings = torch.zeros((num_rows, bucket), dtype=int)
        print(f'Starting compression of bucket {bucket}')
        for i in tqdm(range(math.ceil(num_rows / batch_size))):
            row_start = i * batch_size
            row_stop = (i + 1) * batch_size
            minibatch = same_len_subset[row_start:row_stop]
            for j in range(int(bucket / slice_len)):
                col_start = j * slice_len
                col_stop = (j + 1) * slice_len
                sliced_input_ids = minibatch['input_ids'][:, col_start:col_stop]
                sliced_attentions = minibatch['attention_mask'][:, col_start:col_stop]
                encoded_msgs, _ = compressor.trans_encode(sliced_input_ids, sliced_attentions)
                trans_encodings[row_start:row_stop, col_start:col_stop] = encoded_msgs
        encodings_by_buckets[bucket] = trans_encodings
    return data_by_buckets, encodings_by_buckets


def batched_binarize(data_by_buckets, encodings_by_buckets):
    binarized_by_bucket = {}

    for bucket in sorted(data_by_buckets.keys()):
        msgs = data_by_buckets[bucket]['article']
        detokenized_msgs = data_by_buckets[bucket]['detokenized']
        attentions = data_by_buckets[bucket]['attention_mask']
        encodings = encodings_by_buckets[bucket]

        binarized = []

        for msg, detokenized, attention, encoded in zip(msgs, detokenized_msgs, attentions, encodings):
            if msg != detokenized:
                residuals = get_residuals(msg, detokenized)
            else:
                residuals = {}
            trimmed_encoded_msg = MsgBinaryEncoder.trim_padding(encoded, attention=attention)
            binarized_msg = MsgBinaryEncoder.varint_binarize(trimmed_encoded_msg)
            packed = MsgPacker.binary_pack(binarized_msg, residuals=residuals)
            binarized.append(packed)
        binarized_by_bucket[bucket] = binarized
    return binarized_by_bucket
