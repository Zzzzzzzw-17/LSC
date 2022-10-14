

def sanitized_sentences(sentences, max_len=100):
   
    for sentence in sentences:
        for split_sentence in __split_sentence(sentence, max_len):
            yield split_sentence
          


def __split_sentence(sentence, max_len):
    if len(sentence) <= max_len:
        yield sentence
    else:
        yield sentence[:max_len]
        for sentence_split in __split_sentence(sentence[max_len:], max_len):
            yield sentence_split


def filter_for_words(sentences, target_words):
    sent=[sentence for sentence in sentences]
    sent_new = []
    for sentence in sent:
        if any([target for target in target_words if target in sentence]):
            #print('here', f'{target} in {sentence}')
            #yield sentence
            sent_new.append(sentence)

    return sent_new

def remove_pos_tagging(sentences, target_words):
    for sentence in sentences:
        for word in target_words:
            if not(word.endswith("_nn") or word.endswith("_vb")):
                continue
            san_word = remove_pos_tagging_word(word)
            sentence[:] = [san_word if word == w else w for w in sentence]
        yield sentence


def remove_numbers(sentences):
    sent_new = []
    for sentence in sentences:
        sent_new.append([token for token in sentence if not token.isdecimal()])
    return sent_new


def remove_pos_tagging_word(word: str):
    if word.endswith("_nn") or word.endswith("_vb"):
        return word[:-3]
    return word
