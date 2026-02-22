"""
Text generation module for Italian Lyrics ML.

This is where the magic happens — after training, we use the model to
create new Italian lyrics that never existed before.

How generation works (in plain language):
1. You give the model a starting phrase (e.g., "Amore mio")
2. The model predicts probabilities for what the next word could be
3. We pick one word from those probabilities
4. That word gets added to the sequence
5. Repeat from step 2 until we have enough text

The way we pick the next word matters enormously:
- "Greedy" (always pick the most likely): produces repetitive text
  like "sole sole sole sole sole..."
- "Temperature sampling": adds randomness. Higher temperature = more
  creative but also more nonsensical
- "Top-k sampling": only consider the k most likely words
- "Top-p (nucleus) sampling": only consider words until their combined
  probability reaches p. This adapts — sometimes 3 words cover 90%
  of the probability, sometimes 100 words do.
"""

import torch
import torch.nn.functional as F


def top_k_top_p_filtering(logits, top_k=50, top_p=0.9):
    """
    Filter a probability distribution using top-k and top-p (nucleus) sampling.

    Think of it like this:
    - top_k=50 means "only consider the 50 most likely next words"
    - top_p=0.9 means "only consider words until you've covered 90%
      of the probability"

    Together, these prevent the model from picking extremely unlikely
    words while still allowing creative variety.
    """
    # Top-k: zero out everything except the top k values
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        logits[logits < threshold] = -float('Inf')

    # Top-p: zero out the least likely words until cumulative prob > p
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')

    return logits


def generate_rnn_lstm(model, tokenizer, seed_text, max_length=100,
                      temperature=0.7, top_k=50, top_p=0.9):
    """
    Generate text using an RNN or LSTM model, word by word.

    This is a custom generation loop because RNN/LSTM models don't have
    a built-in generate() method like Transformers do.

    Args:
        model: Trained RNN or LSTM model.
        tokenizer: Converts between words and numbers.
        seed_text: Starting phrase (e.g., "Amore mio").
        max_length: Maximum number of tokens to generate.
        temperature: Controls randomness (0.1=conservative, 1.5=wild).
        top_k: Only consider top k most likely words.
        top_p: Only consider words covering this much probability.

    Returns:
        Generated text as a string.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)
    generated = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(max_length - len(generated)):
            inputs = torch.tensor([generated]).to(device)
            outputs = model(inputs)

            # Get predictions for the next word
            next_token_logits = outputs[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(
                next_token_logits.clone(), top_k=top_k, top_p=top_p
            )

            # Sample from the filtered distribution
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()

            generated.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_transformer(model, tokenizer, seed_text, max_length=100,
                         temperature=0.7, top_k=50, top_p=0.9,
                         repetition_penalty=1.2):
    """
    Generate text using the GPT-2 Transformer model.

    The Transformer has a built-in generate() method with many options.
    The repetition_penalty discourages the model from repeating itself —
    addressing the "sole sole sole sole" problem.

    Args:
        model: Trained GPT-2 model.
        tokenizer: GPT-2 tokenizer.
        seed_text: Starting phrase.
        max_length: Maximum tokens to generate.
        temperature: Randomness control.
        top_k: Top-k filtering.
        top_p: Nucleus sampling threshold.
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty).

    Returns:
        Generated text as a string.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_greedy(model, tokenizer, seed_text, max_length=50):
    """
    Generate text using greedy decoding (always pick the most likely word).

    This is intentionally simple and produces REPETITIVE output.
    We include it as a teaching tool to show WHY sampling strategies matter.
    Compare the output of this function with generate_rnn_lstm() to see
    the difference.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)
    generated = input_ids[0].tolist()

    with torch.no_grad():
        for _ in range(max_length - len(generated)):
            inputs = torch.tensor([generated]).to(device)
            outputs = model(inputs)
            next_token = outputs[0, -1, :].argmax().item()
            generated.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated, skip_special_tokens=True)
