# improve_sentence 미적용
# 재해석 문제 해결
# 다수의 해석이 있는 경우 해결

import pandas as pd
import DAO
from transformers import XLMRobertaTokenizerFast, XLMRobertaForMaskedLM, pipeline
import torch

def get_id(word_replacements):
    return_dic = {}
    for i in list(word_replacements.keys()):
        return_dic[i] = word_replacements[i][0]
    return return_dic


def get_word_replacements(sentence,cur):
    word_replacements = {}
    result = DAO.search_all(cur)
    for i in result:
        if i[0] in sentence:
            synonyms = i[2].split(',')
            synonyms = [synonym.strip() for synonym in synonyms]
            word_replacements[i[0]] = [i[1]]+synonyms
    print(f'DTO에서 출력중 \n {word_replacements}')
    return word_replacements


def check_word_in_sentence(sentence, word_list):
    for word in word_list:
        if word in sentence:
            return word  # 캐치한 신조어를 반환
    return word

def replace_words(sentence, word_replacements):
    sentences = [sentence]
    for word, replacements in word_replacements.items():
        new_sentences = []
        for sent in sentences:
            if word in sent:
                replacement_words = []
                for replacement in replacements[1:]:  # 첫 번째 요소를 제외한 뒷부분만 추가
                    new_sentences.append(sent.replace(word, replacement))
                    replacement_words.append(replacement)
            else:
                new_sentences.append(sent)
        sentences = new_sentences
    return sentences, replacement_words

def mask(sentences, replacement_words):
    masked_sentences = []
    for sent in sentences:
        for word in replacement_words:
            if word in sent:
                word_idx = sent.index(word)
                print(word_idx + len(word))
                print(len(sent))
                replace_mask = sent[word_idx + len(word)]
            mask = sent.replace(replace_mask, '<mask>')
            masked_sentences.append(mask)
            print(masked_sentences)

    return masked_sentences

def tokenized(masked_sentences, tokenizer):
    sequences = []
    for i in range(len(masked_sentences)):
        tokenized_sequence = tokenizer.tokenize(masked_sentences[i])
        sequences.append(tokenized_sequence)
    return sequences

def encoded(sequence, tokenizer):
    encoded_sequences = [tokenizer(sequence[i], add_special_tokens=False)['input_ids'] for i in range(len(sequence))]
    return encoded_sequences

def similarity(decoded_sequence, original_sequence, masked_index, model, tokenizer):
    inputs = tokenizer(decoded_sequence, original_sequence, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prob = torch.softmax(logits, dim=1)
    similarity_score = prob[0, masked_index, tokenizer.mask_token_id].item()
    similarity_score = round(similarity_score, 6) * 100
    return similarity_score

def find_most_natural_sequence(sequence, encoded_sequences, tokenizer, model):
    fill_mask = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)

    most_natural_score = -1
    most_natural_sequence = None

    for idx in range(len(sequence)):
        masked_index = encoded_sequences[idx].index(tokenizer.mask_token_id)
        input_ids = [encoded_sequences[idx]]
        predictions = fill_mask(tokenizer.decode(input_ids[0]))
        top_predictions = predictions[0:3]

        for prediction in top_predictions:
            print(top_predictions)
            print(prediction)
            predicted_token = prediction['token_str']
            decoded_sequence = sequence[idx].replace('<mask>', predicted_token)
            decoded_sequence = decoded_sequence.replace('#', '')
            print(f"교체 후: {decoded_sequence}")

            similarity_score = similarity(decoded_sequence, sequence[idx], masked_index, model, tokenizer)

            if similarity_score > most_natural_score:
                most_natural_score = similarity_score
                most_natural_sequence = decoded_sequence

    return most_natural_sequence

def main(sentence,cur):
    word_replacements = get_word_replacements(sentence,cur)
    return_1= get_id(word_replacements)
    user_input = sentence

    found_word = check_word_in_sentence(user_input, word_replacements.keys())
    if found_word:
        print(f"신조어: {found_word}")
    
        replaced_sentences, replacement_words = replace_words(user_input, word_replacements)
        masked_sentences = mask(replaced_sentences, replacement_words)

        tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
        model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base').eval()

        sequences = tokenized(masked_sentences, tokenizer)
        print(f'main -> tokenized : {sequences}')
        encoded_sequences = encoded(masked_sentences, tokenizer)

        most_natural_sequence = find_most_natural_sequence(masked_sentences, encoded_sequences, tokenizer, model)

    else:
        print("입력한 문장에 설정된 단어가 포함되어 있지 않습니다.")

    print(f'최종 리턴 \n 1번:{return_1} \n 2번{most_natural_sequence}')

    return return_1,most_natural_sequence
