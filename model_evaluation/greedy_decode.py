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

    if src_mask is not None and src_mask.dtype != torch.bool:
        src_mask = src_mask.to(torch.bool)

    memory = model.encoder_layers(src, src_mask)

    tgt_tokens = torch.tensor([[start_token]], device=src.device)

    for _ in range(max_len):
        tgt_mask = model.generate_square_subsequent_mask(tgt_tokens.size(1)).to(src.device)

        output = model.decoder_layers(tgt_tokens, memory, tgt_mask=tgt_mask, memory_mask=src_mask)

        logits = model.output_layer(output[:, -1, :])

        next_token = logits.argmax(dim=-1).item()

        tgt_tokens = torch.cat([tgt_tokens, torch.tensor([[next_token]], device=src.device)], dim=1)

        if next_token == end_token:
            break

    return tgt_tokens.squeeze(0).tolist()  
