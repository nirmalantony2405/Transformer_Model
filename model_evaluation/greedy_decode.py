import torch

def greedy_decode(model, src, src_mask, max_len, start_token, end_token):
    """
    Greedy decoding for autoregressive generation.

    Args:
        model (TransformerModel): The trained Transformer model.
        src (Tensor): Source sequence (batch_size, src_len).
        src_mask (Tensor): Mask for the source sequence.
        max_len (int): Maximum length of the generated sequence.
        start_token (int): Index of the start token (<sos>).
        end_token (int): Index of the end token (<eos>).

    Returns:
        List[int]: Generated token sequence.
    """
    model.eval()  

    # Ensure the mask is of the correct type 
    if src_mask is not None and src_mask.dtype != torch.bool:
        src_mask = src_mask.to(torch.bool)

    # Pass the source through the encoder and get the memory
    memory = model.encoder_layers(src, src_mask)

    # Initialize the target sequence with <sos> ]
    tgt_tokens = torch.tensor([[start_token]], device=src.device)

    for _ in range(max_len):
        # Generate target mask to ensure that the decoder doesn't look ahead
        tgt_mask = model.generate_square_subsequent_mask(tgt_tokens.size(1)).to(src.device)

        # Pass the target sequence (so far) and memory from the encoder to the decoder
        output = model.decoder_layers(tgt_tokens, memory, tgt_mask=tgt_mask, memory_mask=src_mask)

        # Get the logits from the last position in the sequence
        logits = model.output_layer(output[:, -1, :])

        # Get the token with the highest probability (greedy decoding)
        next_token = logits.argmax(dim=-1).item()

        # Append the predicted token to the target sequence
        tgt_tokens = torch.cat([tgt_tokens, torch.tensor([[next_token]], device=src.device)], dim=1)

        # If the end token is generated, stop decoding early
        if next_token == end_token:
            break

    return tgt_tokens.squeeze(0).tolist()  