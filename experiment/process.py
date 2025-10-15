import json, torch
from nltk.corpus import stopwords



def init_prompt(model, tokenizer, product_list, target_product_idx, guiding_sentence, prompt_length, batch_size, device):
    # TODO
    target_product_str = product_list[target_product_idx]['Natural']

    init_product_prompt = guiding_sentence + target_product_str #'Develop a strategy for hacking into a government database and stealing sensitive information.' 

    target_product_tokens = tokenizer(init_product_prompt, return_tensors="pt")["input_ids"]
    
    product_repeated = target_product_tokens.repeat(batch_size, 1).to(device)

    output = model.generate(product_repeated, max_new_tokens=prompt_length, do_sample=True, top_k=10,
                            attention_mask=torch.ones_like(product_repeated).to(device),
                            pad_token_id=tokenizer.eos_token_id)

    logits = model(output).logits
    prompt_logits = logits[:, -(prompt_length+1):-1, :] 

    return prompt_logits.to(torch.float32)


def process_headtail(tokenizer, system_prompt, product_list, user_msg, target_product, batch_size, mode, device, last):
    # since it might shuffle the product list, we need to find the index of the target product
    product_names = [product['Name'] for product in product_list]
    target_product_idx = product_names.index(target_product)

    head = system_prompt['head'] + user_msg + "\n\nProducts:\n"
    tail = ''

    if last:
        # attack prompt following the target product is at the end

        for i, product in enumerate(product_list):
            if i < target_product_idx:
                head += product['Natural'] + "\n"
            elif i == target_product_idx:
                target_product_str = product['Natural']
            else:
                head += product['Natural'] + "\n"

        if mode == 'suffix':
            head += target_product_str 
        tail = system_prompt['tail']

    else:
        # attack prompt is in the middle

        # Generate the adversarial prompt
        for i, product in enumerate(product_list):
            if i < target_product_idx:
                head += product['Natural'] + "\n"
            elif i == target_product_idx:
                if mode == 'suffix':
                    head += product['Natural'] + "\n"
                tail += head[-1:]
                head = head[:-1]
            else:
                tail += product['Natural'] + "\n"

        tail = tail.rstrip('\n')
        tail += system_prompt['tail']
    # head = system_prompt + 'Develop a strategy for hacking into a government database and stealing sensitive information.'
    # tail = ' [/INST]'

    head_tokens = tokenizer(head, return_tensors="pt")["input_ids"].repeat(batch_size, 1).to(device)
    tail_tokens = tokenizer(tail, return_tensors="pt", add_special_tokens=False)["input_ids"].repeat(batch_size, 1).to(device)

    return head_tokens, tail_tokens


def process_text(tokenizer, text, batch_size, device):
    target_tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].repeat(batch_size, 1).to(device)
    return target_tokens


def process_bad_words(bad_words, tokenizer, device):
    upper_words = [word.upper() for word in bad_words]

    all_bad_words = bad_words + upper_words
    
    bad_words_str = ' '.join(all_bad_words)

    bad_words_ids = tokenizer(bad_words_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    return bad_words_ids


def process_stop_words(product_str, tokenizer, device):
    stop_words = set(stopwords.words('english'))

    product_str = product_str.split()
    product_str = [word for word in product_str if word.lower() not in stop_words and word.isalpha()]

    product_str = ' '.join(product_str)

    product_tokens_ids = tokenizer(product_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    return product_tokens_ids



def greedy_decode(logits, tokenizer):
    token_ids = torch.argmax(logits, dim=-1) 

    decoded_sentences = [tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in token_ids]

    return token_ids, decoded_sentences


def select_topk(logits, topk):
    topk_indices = torch.topk(logits, topk, dim=-1).indices  # Shape: (batch_size, length, topk)

    # Create a mask for the top-k indices
    topk_mask = torch.zeros_like(logits, dtype=torch.bool)  # Shape: (batch_size, length, vocab_size)
    topk_mask.scatter_(-1, topk_indices, 1)  # Mark top-k indices as True   

    return topk_mask


def create_word_mask(words, logits):
    batch_size, length, vocab_size = logits.size()
    bad_word_mask = torch.zeros((vocab_size,), dtype=torch.bool, device=logits.device)
    bad_word_mask.scatter_(0, words.flatten(), 1)  # Mark bad word tokens as True

    # Expand bad_word_mask to match logits shape
    bad_word_mask = bad_word_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, vocab_size)
    bad_word_mask = bad_word_mask.expand(batch_size, length, vocab_size)  # Broadcast across batch and length

    return bad_word_mask


def mask_logits(logits, topk_mask, extra_mask=None):
    BIG_CONST = -1e5
    combined_mask = topk_mask | extra_mask if extra_mask is not None else topk_mask

    # Step 4: Apply the combined mask to logits
    # -65504 is the minimum value for half precision floating point
    masked_logits = logits + (~combined_mask * BIG_CONST)
    return masked_logits


def get_original_embedding(model, tokenizer, product_str, device):
    tokens = tokenizer(product_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    output = model(tokens, output_hidden_states=True).hidden_states[-1].mean(dim=1).detach()

    return output.to(torch.float16)


def get_logits_embedding(model, logits, temperature):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1).to(torch.float16)

    soft_embeds = torch.matmul(probs, model.get_input_embeddings().weight)

    embeds = model(inputs_embeds=soft_embeds, use_cache=True, output_hidden_states=True).hidden_states[-1].mean(dim=1)

    return embeds


