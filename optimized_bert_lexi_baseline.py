# imports needed and logging
import gensim
from gensim.models import KeyedVectors
import logging
from smart_open import smart_open
import numpy as np
import os
from nltk.corpus import wordnet as wn
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import collections
from operator import attrgetter, itemgetter
from ConversationClassification import ConversationClassification
from nltk.corpus import stopwords

import rpy2
import pandas as pd
import numpy as np
import psycopg2
from collections import Counter
from collections import OrderedDict

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import postgres_config
import mysql_config
import imp
import copy
imp.reload(mysql_config)

from sqlalchemy import create_engine
import threading
import concurrent.futures
import multiprocessing as mp
from scipy.stats import entropy
import pickle

"""
Grid search on alpha values
Majority Sum Vote
Output first conversation as example
Do KNN on the max(sim n) words in the context that match the sense embedding most
-- Examples don't include target words
-- Unique lemmatized words all together
-- using table 4 words
-- always include embedding of target in sentence
-- if votes are split with uniform distr, choose the one with highest rank
"""


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
logging.info("Starting to run sweeping context WSD")

stop_words = stopwords.words('english')

pos_to_index = {"N":0,"n":0,"v":1, "V":1,"ADV":2, "ADJ":3, "adj":3, "JJR":3, "JJS":3,
                "JJ":3, "NN":0, "NNS":0, "NNP":0, "NNPS":0,"VB": 1, "VBD":1,
                "VBG":1, "VBN":1, "VBP":1, "VBZ":1, "RBS":2, "RB":2, "RBR":2,
                'A':3, 'S':3, 'R':2, 'a':3, 's':3, 'r':2}

wordsense_auth = postgres_config.Authenticator('WordSense')
ws_engine = create_engine(wordsense_auth.connectionString)

childesdb_auth  = mysql_config.Authenticator('ec2')
cdb_engine = create_engine(childesdb_auth.connectionString)
derived_tokens = pd.read_csv('derived_tokens_2020_01_28.csv')

from bert_serving.client import BertClient
print("Starting client")
bc = BertClient(check_length=False)
print("done client")

Lem = WordNetLemmatizer()
WINDOW_SIZE = 6
KNNs = [3,4] # 3,4 #range(3,6) # range(2,6) number of nearest neighbours

model = KeyedVectors.load_word2vec_format('../crawl-300d-2M.vec')
model["n't"] = model["not"]
word_pos_bleu_senses_2 = pickle.load(open("word_pos_bleu_senses_2_n.pickle", "rb"))

def get_embedding(word):
    return model[word]

def get_bert_embedding(input, embed_type="word"):
    if embed_type == "sentence":
        return bc.encode(input)
    else:
        list_word_embeddings = bc.encode(input)
        return [np.mean(sent, axis=0)
                for sent in list_word_embeddings]
w_pos_to_synset_mean = {}

def get_tagged_synsets(target_word, childes_postag, sense_to_freq):

    if (target_word+childes_postag) in w_pos_to_synset_mean:
        return w_pos_to_synset_mean[target_word+childes_postag]

    all_synsets = wn.synsets(target_word)
    synsets = []

    for s in all_synsets:
        if (s.pos() in pos_to_index) and (childes_postag in pos_to_index) \
                         and (pos_to_index[s.pos()] == pos_to_index[childes_postag]):
            synsets.append(s)

    lexi_def_words = [(s, nltk.pos_tag((nltk.word_tokenize(s.definition())))) for s in synsets] # indexed similarly to synsets
    lexi_syn_means_vec = [(s, get_mean_context_vector(c, target_word, "SENSE")) for s, c in lexi_def_words]

    bert_def_embeddings = get_bert_embedding([s.definition() for s in synsets])
    bert_syn_means_vec = []
    # find knn mean of example contexts
    for i in range(len(synsets)):
        sense = synsets[i]
        if sense.examples():
            bert_ex_embedding = np.mean(get_bert_embedding(sense.examples()), axis=0)
            bert_syn_means_vec.append((sense,(np.mean([bert_def_embeddings[i]] + [bert_ex_embedding],axis=0), [])))
        else:
           bert_syn_means_vec.append((sense, (bert_def_embeddings[i], [])))
        mean_embedding = lexi_syn_means_vec[i][1][0]
        used_words = lexi_syn_means_vec[i][1][1]
        example_tokenized = nltk.word_tokenize(" ".join(sense.examples()))
        to_calc = []
        for ex_word in example_tokenized:
            if ex_word[0] not in used_words:
                to_calc.append(ex_word)

        syn_example_words = [] if sense.examples() == [] else nltk.pos_tag(to_calc)

        if syn_example_words: # mean_embedding
            syn_ex_mean_knn_contexts, used_ex = get_mean_context_vector(syn_example_words, target_word,"SENSE", syn_vec=mean_embedding, knns=[3]) # 3
            # new mean:
            sum_def_embedding = np.array(mean_embedding) * len(used_words)
            sum_ex_knn_emb = np.array(syn_ex_mean_knn_contexts) * len(used_ex)

            total_words = list(set(used_words).union(used_ex))
            common_words = list(set(used_words).intersection(set(used_ex)))
            common_embeddings = np.zeros(sum_def_embedding.shape[0]) if\
                not common_words else np.sum([get_embedding(w) for w in common_words], axis=0)

            total_sense_mean = ((sum_ex_knn_emb + sum_def_embedding -
                                 common_embeddings) / len(total_words))
            lexi_syn_means_vec[i] = (sense,(list(total_sense_mean), total_words))

    w_pos_to_synset_mean[target_word+childes_postag] = {"lexi":{"synsets":synsets,
                                                                "syn_means_vec": lexi_syn_means_vec},
                                                        "bert": {"synsets":synsets,
                                                                 "syn_means_vec": bert_syn_means_vec}}

    return {"lexi":{"synsets":synsets, "syn_means_vec": lexi_syn_means_vec},
            "bert": {"synsets":synsets,"syn_means_vec": bert_syn_means_vec}}


def get_closest_sense(contexts, main_sent, target_word,
                      childes_postag, synsets, syn_means_vec,
                      sense_to_freq, model="lexi"):
        """

        :param contexts: List of contexts from the window
        :param main_sent: Main sentence
        :param target_word: Lemmatized Target Word
        :param childes_postag:the pos tag
        :param synsets: List of synsets from WN
        :param syn_means_vec: Synset embeddings
        :param sense_to_freq: {sense: Probability, ...}
        :param model: LEXI OR BERT based on embeddings
        :return: Best Conversation
        """
        cos_similarity = []
        try:
            if not childes_postag in pos_to_index:
                return None # only classify convos that have a valid POS tag
            prior_to_alpha_to_context_syn_pos_prob = {p : {a:[] for a in alpha_range} for p in ["wn","childes"]} # P(childes_po | syn_pos) probabilities
            # COMPUTE the probability
            prior_to_wn_freq_pos = {p: synsets[0] for p in ["wn", "childes"]}
            total_senses = len(synsets)
             # sense_to_freq_max_name = max(sense_to_freq)
            max_sense = 0

            for i in range(total_senses):
                #context_syn_pos_prob.append((synsets[i], ((i+1)/total_senses)**-1))
                for alpha in alpha_range:
                    for prior in prior_to_alpha_to_context_syn_pos_prob:
                        if prior == "wn":
                            p_sense = (total_senses - (i*1))
                        else:
                            p_sense = sense_to_freq[synsets[i].name()] if synsets[i].name() in sense_to_freq else 0
                            try:
                                p_sense /= float(sum(sense_to_freq.values()))
                            except:
                                p_sense = 1
                            # Set the MFS depending on the Childes data
                            if p_sense > max_sense:
                                max_sense = p_sense
                                prior_to_wn_freq_pos[prior] = synsets[i]
                        if alpha == 0: # don't take prio into consideration
                            prior_to_alpha_to_context_syn_pos_prob[prior][alpha].append((synsets[i], 1))
                        else:
                            prior_to_alpha_to_context_syn_pos_prob[prior][alpha].append((synsets[i], p_sense**alpha))

            # obtain a list of word embeddings for this sentence
            sum_contexts = sum(contexts, []) # syn_vec
            for s,vec_cont in syn_means_vec:
                syn_vec = vec_cont[0]
                syn_contexts = vec_cont[1]

                if model=="bert":
                    str_context, target_indeces = get_tokenized(sum_contexts, target_word)
                    list_word_embeddings = bc.encode([str_context])[0]
                    
                    target_embedding = []
                    for embedding_index in target_indeces:
                        if embedding_index + 1 < 25:
                            target_embedding.append(list_word_embeddings[embedding_index + 1])
                    if target_embedding == []:
                        target_embedding = list_word_embeddings

                    context_mean_vec, target_cont_used = np.mean(target_embedding, axis=0), [] #  include target for sentence
                else:
                    context_mean_vec, target_cont_used = get_mean_context_vector(sum_contexts, target_word, "CONTEXT", syn_vec=get_embedding(target_word), knns=KNNs) # include target for sentence

                conversation = " ".join([w[0] if not w[0] == "CLITIC" else "" for w in sum_contexts])

                # Get the definition and examples
                if target_cont_used == [] and model == "lexi":
                    print("Context words not found for conversation: {}".format(conversation))
                    return None
                self_similarity = cosine_vec_similarirty(context_mean_vec, context_mean_vec)
                cos_similarity.append((s, cosine_vec_similarirty(context_mean_vec, syn_vec) / self_similarity, syn_contexts))

            prior_to_alpha_to_cos_pos_prob = {p: {a:[] for a in alpha_range} for p in ["wn", "childes"]}
            prior_to_alpha_to_sense_class_sorted = {p: {a:[] for a in alpha_range} for p in ["wn", "childes"]}
            prior_to_alpha_to_most_sim_conv = {p:{} for p in ["wn", "childes"]} # alpha to the correct classification
            for prior in prior_to_alpha_to_cos_pos_prob:
                for alpha in prior_to_alpha_to_cos_pos_prob[prior]:
                    for i in range(len(cos_similarity)):
                        assert(cos_similarity[i][0].name() == prior_to_alpha_to_context_syn_pos_prob[prior][alpha][i][0].name())
                        prior_to_alpha_to_cos_pos_prob[prior][alpha].append(ConversationClassification(cos_similarity[i][0], # the sense
                                             cos_similarity[i][2], # the sense context words used
                                             cos_similarity[i][1] * prior_to_alpha_to_context_syn_pos_prob[prior][alpha][i][1],  # cos_similarity
                                             target_cont_used, # target context
                                             childes_postag, # the target pos,
                                             target_word, # the target word
                                             conversation, # the conversation we're studying
                                             prior_to_wn_freq_pos[prior], main_sent)
                                             )
                        prior_to_alpha_to_sense_class_sorted[prior][alpha].append((cos_similarity[i][1] * prior_to_alpha_to_context_syn_pos_prob[prior][alpha][i][1],cos_similarity[i][0]))

                    most_similar_conv = max(prior_to_alpha_to_cos_pos_prob[prior][alpha],key= lambda o: o.cos_similarity)
                    prior_to_alpha_to_sense_class_sorted[prior][alpha] = sorted(prior_to_alpha_to_sense_class_sorted[prior][alpha], reverse=True)[:2]
                    most_similar_conv.top_uniform_tags = [sc[1].name() for sc in prior_to_alpha_to_sense_class_sorted[prior][alpha]]
                    prior_to_alpha_to_most_sim_conv[prior][alpha] = most_similar_conv

            return prior_to_alpha_to_most_sim_conv
        except Exception as e:
           print("Error: get_closest_sense() ", e)
           return None

def window_classification(args):
    """
    Construct a conversation window and classify the correct tense
    """
    w_info, utterance_index, target_sentence, target_word,target_pos_tag,\
    target_sense_list, target_sense_str_list,\
    model_to_synsets,token_record, sense_to_freq = args

    w_index,w_size = w_info
    window_index = w_index + 1
    start_context = utterance_index - window_index
    end_context = (w_size - window_index) + utterance_index + 1
    assert (w_size + 1 == end_context - start_context)
    if start_context < 0:
        return None
    context = []
    for indeces in range(start_context, end_context):
        query = 'SELECT  u.id from utterance u where '+\
        'u.utterance_order={} and u.transcript_id={} AND '.format(indeces,
                                                    token_record['transcript_id'])+\
        '(u.corpus_id=204 OR u.corpus_id=49)'
        prev_utterances_df = pd.read_sql_query(query,cdb_engine)
        try:
            prev_utterance_id = prev_utterances_df.to_dict('records')[0]['id']
        except:
            # index out of range
            print("Window index out of range")
            continue
        utterance = derived_tokens.loc[
        (derived_tokens.transcript_id == token_record['transcript_id']) &
        (derived_tokens.utterance_id == prev_utterance_id)
        ]
        context.append([(w,pos) for w,pos in zip(utterance.gloss_with_replacement, utterance.part_of_speech)])

    prior_to_alpha_to_convo = get_closest_sense(context, target_sentence,
                              target_word,target_pos_tag,
                              model_to_synsets["lexi"]["synsets"],
                              model_to_synsets["lexi"]["syn_means_vec"],
                              sense_to_freq, model="lexi")

    prior_to_alpha_to_bert_convo = get_closest_sense(context, target_sentence,
                              target_word,target_pos_tag,
                              model_to_synsets["bert"]["synsets"],
                              model_to_synsets["bert"]["syn_means_vec"],
                              sense_to_freq, model="bert")
    if prior_to_alpha_to_convo is None:
        print("Conversation is None")
        return None
    for prior in prior_to_alpha_to_convo:
        for alpha in alpha_range:
            if prior_to_alpha_to_convo[prior][alpha] is None:
                continue
            prior_to_alpha_to_convo[prior][alpha].window_size = "-{} : +{}".format(window_index, w_size - window_index)
            prior_to_alpha_to_convo[prior][alpha].main_sent_index = utterance_index
            prior_to_alpha_to_convo[prior][alpha].gt_sense_list = target_sense_list
            prior_to_alpha_to_convo[prior][alpha].gt_str_sense_list = target_sense_str_list
            prior_to_alpha_to_convo[prior][alpha].bert_sense = prior_to_alpha_to_bert_convo[prior][alpha].classified_sense
    return prior_to_alpha_to_convo

def classify_tag_distributed(tag, tied_tags, sense_to_freq, speaker="BOTH"):
    """
    Retrieve the tag for the given speaker
    """
    token_record = derived_tokens.loc[derived_tokens.id == tag['token_id']].to_dict('records')[0]
    target_sense_str_list = [t['wordnet_sense'] for t in tied_tags]
    target_sense_list = []
    target_pos_tag = tag['part_of_speech']
    target_word = tag['lemma']
    utterance_id = token_record['utterance_id']
    transcript_id = token_record['transcript_id']
    target_utterance = derived_tokens.loc[
    (derived_tokens.transcript_id == token_record['transcript_id']) &
    (derived_tokens.utterance_id == token_record['utterance_id'])
    ]
    target_sentence = [(w,pos) for w,pos in zip(target_utterance.gloss_with_replacement, target_utterance.part_of_speech)]

    query = 'SELECT  u.utterance_order, u.speaker_code from utterance u where '+\
            'u.id={} and u.transcript_id={} AND '.format(token_record['utterance_id'],
                                                        token_record['transcript_id'])+\
            '(u.corpus_id=204 OR u.corpus_id=49)'
    utterances_df = pd.read_sql_query(query,cdb_engine)
    utterance_index = utterances_df.to_dict('records')[0]['utterance_order']
    speaker_code = utterances_df.to_dict('records')[0]['speaker_code']
    if speaker == "BOTH" or (speaker == speaker_code) or (speaker == "PARENT" and not speaker_code == "CHI"):
        # Find closest sense
        try:
            model_to_synsets = get_tagged_synsets(target_word, target_pos_tag, sense_to_freq)
            for sense in model_to_synsets["lexi"]["synsets"]:
                if sense.name() in target_sense_str_list:
                    target_sense_list.append(sense)
                    break
        except Exception as e:
            print("Could not get synsets", e)
            return None
        if not sense in model_to_synsets["lexi"]["synsets"] or not target_sense_list:
            print("Sense not included in synsets")
            return None

        args = [(WINDOW_SIZE // 2, WINDOW_SIZE), utterance_index,
              target_sentence, target_word,target_pos_tag,
              target_sense_list, target_sense_str_list,
              model_to_synsets,
              token_record, sense_to_freq]
        prior_to_alpha_to_convo_classification = window_classification(args)
        if prior_to_alpha_to_convo_classification is None:
            print("Winodw classification returned None")
            return None
        for prior in prior_to_alpha_to_convo_classification:
            for alpha in alpha_range:
                if prior_to_alpha_to_convo_classification[prior][alpha] is None:
                    continue
                prior_to_alpha_to_convo_classification[prior][alpha].id = str(transcript_id) + "_" + str(utterance_id)
        return prior_to_alpha_to_convo_classification
    else:
        # Wrong speaker
        return None

def read_input(tag_set_csv, pos_spec, epy_upper, alpha_range, prior_list, SPEAKER_TYPE="BOTH"):

    """
    Read every word in our csv file and classify them based on entropy
    """
    logging.info("Starting to read files...this may take a while")

    alpha_to_all_conversation_results = {p:{a:[] for a in alpha_range} for p in prior_list}
    seen_token_ids = set()
    tag_set_df = pd.read_csv(tag_set_csv)

    lxb_matches = {p : {a: 0 for a in alpha_range} for p in prior_list}
    mfs_matches = {p : {a: 0 for a in alpha_range} for p in prior_list}
    poly_tags = {p : {a: 0 for a in alpha_range} for p in prior_list}
    lmxb_matches = {p : {a: 0 for a in alpha_range} for p in prior_list}
    multi_tags= {p : {a: 0 for a in alpha_range} for p in prior_list}
    synonym_tags = {p : {a: 0 for a in alpha_range} for p in prior_list}
    mfs_synonym_tags = {p : {a: 0 for a in alpha_range} for p in prior_list}
    total_convos = {p : {a: 0 for a in alpha_range} for p in prior_list}
    bert_matches = {p : {a: 0 for a in alpha_range} for p in prior_list}
    all_tokens = 0
    lemma_to_alpha_to_acc = {p: {a: {} for a in alpha_range} for p in prior_list}
    set_words = list(set(tag_set_df.lemma))
    word_to_sense_counter = {} #{w : set() for w in set_words}
    w_to_pos = {}
    w_to_list_index = {}
    all_words_pos = 0
    alpha_to_finalstring = {p :{a:"" for a in alpha_range} for p in prior_list}

    for w in set_words:
        dict_tokens = tag_set_df.loc[tag_set_df.lemma == w].to_dict('records')
        pos_set = set()
        for token in dict_tokens:
            word_pos = w + '_' + token['part_of_speech']
            pos_set.add(token['part_of_speech'])
            if not word_pos in word_to_sense_counter:
                word_to_sense_counter[word_pos] = Counter()

            if not token['wordnet_sense'] in {'idk','other_meanings'}:
                word_to_sense_counter[word_pos][token['wordnet_sense']] += 1

        for pos in pos_set:
            word_pos = w + '_' + pos
            all_words_pos += 1
            total = float(sum(word_to_sense_counter[word_pos].values()))
            relative_freq_list = []
            for key in word_to_sense_counter[word_pos]:
                relative_freq_list.append(word_to_sense_counter[word_pos][key]/ total)

            enpy = entropy(relative_freq_list, base=2)

            if pos==POS_SPEC and ((epy_upper == -1 and enpy == 0) or (epy_upper - 1 <= enpy < epy_upper and enpy > 0)):
                if not w in w_to_pos:
                    w_to_pos[w] = set()
                w_to_pos[w].add(pos)

                poly_df = tag_set_df.loc[tag_set_df.lemma == w]
                poly_df = poly_df.loc[poly_df.part_of_speech == pos]
                w_to_list_index[word_pos] = {"tokens": poly_df.to_dict('records', into=OrderedDict), "read_idx":0}
                all_tokens += len(w_to_list_index[word_pos]["tokens"])

    completed = 0
    top_epy_word_pos = len(w_to_list_index)
    print(epy_upper, all_words_pos, " --> ", top_epy_word_pos)
    all_completed = 0 # all conversations regardless of speaker
    while ((completed < 500) and (all_completed < all_tokens)):
        for word_pos in w_to_list_index:
            read_idx = w_to_list_index[word_pos]["read_idx"]

            for prior in prior_list:
                for alpha in alpha_range:
                    if not word_pos in lemma_to_alpha_to_acc[prior][alpha]:
                        lemma_to_alpha_to_acc[prior][alpha][word_pos] = {"acc":0, "mfs": 0, "poly":0, "total":0, "total_mfs":0, "total_poly":0}
            try:
                tag = w_to_list_index[word_pos]["tokens"][read_idx]
                w_to_list_index[word_pos]["read_idx"] +=1
                all_completed += 1
            except:
                continue
            if tag['token_id'] in seen_token_ids:
                continue
            else:
                seen_token_ids.add(tag['token_id']) # TODO race condition

            #tied_tags = poly_df.loc[poly_df.token_id == tag['token_id']].to_dict('records')
            tied_tags = [tag]
            word_to_sense_counter[word_pos][tag['wordnet_sense']] -= 1
            prior_to_alpha_to_convo_classification = classify_tag_distributed(tag, tied_tags, word_to_sense_counter[word_pos],SPEAKER_TYPE)
            word_to_sense_counter[word_pos][tag['wordnet_sense']] += 1

            if prior_to_alpha_to_convo_classification is None:
                continue
            else:
                completed += 1
            for prior in prior_list:
                for alpha in alpha_range:
                    convo_classification = prior_to_alpha_to_convo_classification[prior][alpha]
                    if convo_classification is None:
                        continue

                    alpha_to_all_conversation_results[prior][alpha].append(convo_classification)
                    total_convos[prior][alpha] += 1

                    if convo_classification.classified_sense.name() in convo_classification.gt_str_sense_list:
                        lxb_matches[prior][alpha] += 1# lexical chaining matches
                        synonym_tags[prior][alpha] += 1
                    elif word_pos in word_pos_bleu_senses_2 and\
                                    convo_classification.classified_sense.name()  in word_pos_bleu_senses_2[word_pos]:
                        synonym_tags[prior][alpha] += 1 if len(word_pos_bleu_senses_2[word_pos][convo_classification.classified_sense.name()].intersection(set(convo_classification.gt_str_sense_list))) > 0 else 0

                    lmxb_matches[prior][alpha] += 1 if convo_classification.classified_sense.name() in convo_classification.gt_str_sense_list \
                                        and convo_classification.classified_sense.name() == convo_classification.wn_freq_pos.name()  else 0 # lexical chaining matches

                    if convo_classification.bert_sense.name() in convo_classification.gt_str_sense_list:
                        bert_matches[prior][alpha] += 1# lexical chaining matches

                    if convo_classification.wn_freq_pos.name() in convo_classification.gt_str_sense_list:
                        mfs_matches[prior][alpha] += 1
                        mfs_synonym_tags[prior][alpha] += 1
                    elif word_pos in word_pos_bleu_senses_2 and\
                                    convo_classification.wn_freq_pos.name() in word_pos_bleu_senses_2[word_pos]:
                        mfs_synonym_tags[prior][alpha] +=  1 if len(word_pos_bleu_senses_2[word_pos][convo_classification.wn_freq_pos.name()].intersection(set(convo_classification.gt_str_sense_list))) > 0 else 0

                    poly_tags[prior][alpha] += 0 if convo_classification.wn_freq_pos.name() in convo_classification.gt_str_sense_list else 1

                    multi_tags[prior][alpha] += 0 if len(set(convo_classification.top_uniform_tags).intersection(set(convo_classification.gt_str_sense_list))) == 0 else 1


                    if  convo_classification.classified_sense.name() in convo_classification.gt_str_sense_list:
                        lemma_to_alpha_to_acc[prior][alpha][word_pos]["acc"] += 1
                        lemma_to_alpha_to_acc[prior][alpha][word_pos]["mfs"] += 1 if convo_classification.classified_sense.name() == convo_classification.wn_freq_pos.name()  else 0
                        lemma_to_alpha_to_acc[prior][alpha][word_pos]["poly"] += 1 if not convo_classification.classified_sense.name() == convo_classification.wn_freq_pos.name()  else 0
                    lemma_to_alpha_to_acc[prior][alpha][word_pos]["total_mfs"] += 1 if convo_classification.wn_freq_pos.name() in convo_classification.gt_str_sense_list else 0
                    lemma_to_alpha_to_acc[prior][alpha][word_pos]["total"] += 1
                    lemma_to_alpha_to_acc[prior][alpha][word_pos]["total_poly"] = lemma_to_alpha_to_acc[prior][alpha][word_pos]["total"] - lemma_to_alpha_to_acc[prior][alpha][word_pos]["total_mfs"]

                    # print(word_pos, "Accuracy: {}/{} = {} | POLY: {}/{} = {} | MFS: {}/{} = {} | ".format(
                    #     lemma_to_alpha_to_acc[alpha][word_pos]["acc"], lemma_to_alpha_to_acc[alpha][word_pos]["total"],
                    #     100 * lemma_to_alpha_to_acc[alpha][word_pos]["acc"] / lemma_to_alpha_to_acc[alpha][word_pos]["total"],
                    #     lemma_to_alpha_to_acc[alpha][word_pos]["poly"], lemma_to_alpha_to_acc[alpha][word_pos]["total_poly"],
                    #     100 * lemma_to_alpha_to_acc[alpha][word_pos]["poly"] /  (lemma_to_alpha_to_acc[alpha][word_pos]["total_poly"] if lemma_to_alpha_to_acc[alpha][word_pos]["total_poly"] > 0 else 1),
                    #     lemma_to_alpha_to_acc[alpha][word_pos]["mfs"], lemma_to_alpha_to_acc[alpha][word_pos]["total_mfs"],
                    #     100 * lemma_to_alpha_to_acc[alpha][word_pos]["mfs"] /  (lemma_to_alpha_to_acc[alpha][word_pos]["total_mfs"] if lemma_to_alpha_to_acc[alpha][word_pos]["total_mfs"] > 0 else 1)),
                    #     word_to_sense_counter[word_pos])
                    # print(alpha,epy_upper,total_convos[alpha])
        for prior in prior_list:
            for alpha in alpha_range:
                if total_convos[prior][alpha] > 0:
                    print("Lexical Chaining Bayesian: {} out of {} = %{}\n"
                          "Bert Embedding Bayesian: {} out of {} = %{}\n"
                          "Most Frequent Sense Model: {} out of {} = %{}\n"
                          "Synonym Lexical Chaining: {} out of {} = %{}\n"
                          "Synonym Most Frequent Sense: {} out of {} = %{}\n"
                          "Top 2 Accuracy: {} out of {} = %{}\n".format
                        (lxb_matches[prior][alpha],total_convos[prior][alpha],(lxb_matches[prior][alpha] / total_convos[prior][alpha]) * 100,
                         bert_matches[prior][alpha],total_convos[prior][alpha],(bert_matches[prior][alpha] / total_convos[prior][alpha]) * 100,
                        mfs_matches[prior][alpha], total_convos[prior][alpha],(mfs_matches[prior][alpha] / total_convos[prior][alpha]) * 100,
                        synonym_tags[prior][alpha], total_convos[prior][alpha],(synonym_tags[prior][alpha] / total_convos[prior][alpha]) * 100,
                        mfs_synonym_tags[prior][alpha], total_convos[prior][alpha],(mfs_synonym_tags[prior][alpha] / total_convos[prior][alpha]) * 100,
                        multi_tags[prior][alpha], total_convos[prior][alpha],(multi_tags[prior][alpha] / total_convos[prior][alpha]) * 100,
                         ))

                    alpha_to_finalstring[prior][alpha] = "{},{},{},{},{},{},{},{},".format(alpha, epy_upper,
                                                              lxb_matches[prior][alpha] / total_convos[prior][alpha],
                                                              bert_matches[prior][alpha] / total_convos[prior][alpha],
                                                              mfs_matches[prior][alpha] / total_convos[prior][alpha],
                                                              synonym_tags[prior][alpha] / total_convos[prior][alpha],
                                                              mfs_synonym_tags[prior][alpha] / total_convos[prior][alpha],
                                                              multi_tags[prior][alpha] / total_convos[prior][alpha])
            
    return (alpha_to_all_conversation_results, alpha_to_finalstring)

def get_mean_context_vector(contexts, target_word, type, syn_vec=None, knns = [2]):
    """
    Return a tuple of (mean_context_vector, context_words_used)
    :param contexts:
    :param target_word: "" if context, target_word otherwise
    :param type: either 'SENSE' or 'CONTEXT'
    :param syn_vec_cont:
    :return:
    """
    word_embeddings = []
    used_words = []
    seen_words = set()
    sorted_sim_context_subset = []
    target_embedding =  [] # get_embedding(target_word)

    for tagged_word in contexts:
        spaced_underscore_phrase = " ".join(tagged_word[0].split("_")) # childes combines phrases
        word_tokenized = gensim.utils.simple_preprocess(spaced_underscore_phrase)
        if not word_tokenized:
            continue

        curr_pos_tag = tagged_word[1]
        for curr_word in word_tokenized:

            lemword = Lem.lemmatize(curr_word) # turn the plural into singular
            if curr_pos_tag in {"V", "VB","VBD", "VBG", "VBN", "VBP", "VBZ", "v"}:
                lemword = Lem.lemmatize(curr_word, "v")

            if ((lemword == target_word) or (target_word in lemword)) and type == "SENSE": # TOFIX
                # only exclude target for sense embeddings, keep target for contenxts
                continue
            if type == "SENSE":
                curr_word = lemword
            if (curr_word not in seen_words) and (curr_word not in stop_words) and\
                    (curr_pos_tag in pos_to_index):# and (lemword not in seen_words)
                try:
                    embedding_vector = get_embedding(curr_word)
                    word_embeddings.append(embedding_vector)
                    if syn_vec is not None:
                        sorted_sim_context_subset.\
                            append((curr_word, embedding_vector, cosine_vec_similarirty(embedding_vector, syn_vec)))
                    seen_words.add(curr_word)
                except Exception as e:
                    print("error : ", e, ", ",
                          curr_word,' word embedding not found')

    # average the best KNN word embeddings into a feature vector
    try:
        sorted_sim_context_subset.sort(key = itemgetter(2), reverse = True) # list of all words and their vectors
        # get max average varying K CONTEXTS and add them to the target
        mean_KNN_context = [] # (mean_embedding, cos_sym, [words]) mean sim of every K, KNNs and the syn vec
        if syn_vec is not None:
            # Calculate the sim between average embedding & syn_vec
            for k in knns:
                if k < len(sorted_sim_context_subset):
                    top_k_context = sorted_sim_context_subset[:k]
                    mean_embedding = list(np.mean([w[1] for w in top_k_context] + target_embedding, axis=0))
                    mean_KNN_context.append(
                                (mean_embedding, cosine_vec_similarirty(mean_embedding, syn_vec),
                                 [w[0] for w in top_k_context] + []))
            if mean_KNN_context:
                chosen_context_knn = max(mean_KNN_context, key=itemgetter(1))
                sent_features, used_words = chosen_context_knn[0],\
                                            chosen_context_knn[2]

        if mean_KNN_context == []: # if we're not getting cont subset or did so unsuccessfully
            used_words = list(seen_words)
            sent_features = np.mean(word_embeddings, axis=0)
        return sent_features, used_words
    except Exception as e:
        print("Exception {} (Could not calculate sentence mean)".format(e))
        return [], used_words

def get_tokenized(contexts, target_word):
    tokenized = []
    target_indeces = []
    for tagged_word in contexts:
        spaced_underscore_phrase = " ".join(tagged_word[0].split("_")) # childes combines phrases
        word_tokenized = gensim.utils.simple_preprocess(spaced_underscore_phrase)
        if not word_tokenized:
            continue

        curr_pos_tag = tagged_word[1]
        for curr_word in word_tokenized:
            lemword = Lem.lemmatize(curr_word) # turn the plural into singular
            if curr_pos_tag in {"V", "VB","VBD", "VBG", "VBN", "VBP", "VBZ", "v"}:
                lemword = Lem.lemmatize(curr_word, "v") # get the base tense of the verb TODO

            if "'" in lemword:
                continue
            tokenized.append(lemword)

    for i in range(len(tokenized)):
        if tokenized[i] == target_word or target_word in tokenized[i]:
            target_indeces.append(i)

    return (" ".join(tokenized), target_indeces)


def cosine_vec_similarirty(vec1, vec2):
    return np.dot(vec1, vec2)/\
                               (np.linalg.norm(vec1)* np.linalg.norm(vec2))

def sense_keys(sense):
    """
    For Senseval evaluation
    :param word:
    :return:
    """
    sense_key = [i.key() for i in sense.lemmas()]
    return sense_key

all_sets = [("majority", "majority_tag.csv")] # [("ties", "tie_inclusion_tags.csv")] #("majority", "majority_tag.csv")]

pos_to_prior_to_alpha_epy = {"n":{
                                "prior_list":["wn", "childes"],
                                "alpha_ranges": [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                                "entropy_ranges": [-1,1,2,3],
                                "file": {}

                             },
                             "v": {

                                "prior_list":["wn", "childes"],
                                "alpha_ranges": [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                                "entropy_ranges": [-1,1,2,3,4],
                                "file": {}
                             }
                             }

dir = "berkley_ws_analysis_results"
if not os.path.exists(dir):
    os.mkdir(dir)
SPEAKER_TYPE = "MOT" # or CHI
word_to_embedding = {}
for POS_SPEC in pos_to_prior_to_alpha_epy:

    results_csv = "results_epy_alpha_accuracies_{}_bert_lexi_bert_cbow_{}.csv".format(POS_SPEC, SPEAKER_TYPE)
    results_csv = open(dir + "/" +  results_csv,"w+")
    results_csv.write("Accuracy,Entropy,Alpha,Word Class,Model\n")
    pos_to_prior_to_alpha_epy[POS_SPEC]["results_file"] = results_csv

    for PRIOR_TYPE in pos_to_prior_to_alpha_epy[POS_SPEC]["prior_list"]: #  "childes"
        epy_alpha_csv = "epy_alpha_accuracies_{}_bert_lexi_{}_bert_cbow_{}.csv".format(POS_SPEC, PRIOR_TYPE, SPEAKER_TYPE)
        epy_alpha_file = open(dir + "/" + epy_alpha_csv,"w+")
        epy_alpha_file.write("alpha,epy,Lexical Chaining Acc,Bert Bayesian Acc,MFS Acc,Synonym LC Acc,Synonym MFS Acc,Top 2 Acc,"
                             "% Correct Poly,% Correct MFS,Bert % Correct Poly,Bert % Correct MFS,Total Polysemous Tags %,Total MFS Tags %\n")
        pos_to_prior_to_alpha_epy[POS_SPEC]["file"][PRIOR_TYPE] = epy_alpha_file


    epy_ranges = pos_to_prior_to_alpha_epy[POS_SPEC]["entropy_ranges"]
    for entropy_num in epy_ranges:
        for category, tag_set_csv in all_sets:
            alpha_range = pos_to_prior_to_alpha_epy[POS_SPEC]["alpha_ranges"]
            alpha_to_all_conv_results, alpha_to_result_string = read_input(tag_set_csv, POS_SPEC, entropy_num, alpha_range, pos_to_prior_to_alpha_epy[POS_SPEC]["prior_list"], SPEAKER_TYPE)

            for PRIOR_TYPE in pos_to_prior_to_alpha_epy[POS_SPEC]["prior_list"]: #  "childes"
                for alpha in alpha_range:
                    all_conv_results = alpha_to_all_conv_results[PRIOR_TYPE][alpha]
                    print(category, tag_set_csv)
                    
                    len_convo = len(all_conv_results)

                    lxb_matches = 0
                    bert_matches = 0
                    lexi_mfs = 0
                    mfs_matches = 0
                    poly_tags = 0

                    multi_tags = 0
                    bert_lexi_mfs = 0
                    

                    for conv in all_conv_results:
                        if conv.classified_sense.name() in conv.gt_str_sense_list:
                            lxb_matches += 1 # lexical chaining matches
                            if conv.wn_freq_pos.name() == conv.classified_sense.name():
                                lexi_mfs += 1
                        if conv.wn_freq_pos.name() in conv.gt_str_sense_list:
                            mfs_matches += 1
                        poly_tags += 0 if conv.wn_freq_pos.name() in conv.gt_str_sense_list else 1

                        multi_tags += 0 if len(set(conv.top_uniform_tags).intersection(set(conv.gt_str_sense_list))) == 0 else 1
                        # outputFile.write(str(conv)+"\n")

                        if conv.bert_sense.name() in conv.gt_str_sense_list:
                            bert_matches += 1
                            if conv.wn_freq_pos.name() == conv.bert_sense.name():
                                bert_lexi_mfs += 1

                    
                    try:
                        print(alpha_to_result_string[PRIOR_TYPE][alpha] + "{},{},{},{},{},{}\n".format(
                                             (lxb_matches - lexi_mfs) / poly_tags,
                                             lexi_mfs / (len_convo - poly_tags),
                                             (bert_matches - bert_lexi_mfs) / poly_tags,
                                             bert_lexi_mfs / (len_convo - poly_tags),
                                             (poly_tags / len_convo),
                                              ((len_convo - poly_tags)/len_convo)))
                        pos_to_prior_to_alpha_epy[POS_SPEC]["file"][PRIOR_TYPE].write(alpha_to_result_string[PRIOR_TYPE][alpha] + "{},{},{},{},{},{}\n".format(
                                             (lxb_matches - lexi_mfs) / poly_tags,
                                             lexi_mfs / (len_convo - poly_tags),
                                             (bert_matches - bert_lexi_mfs) / poly_tags,
                                             bert_lexi_mfs / (len_convo - poly_tags),
                                             (poly_tags / len_convo),
                                              ((len_convo - poly_tags)/len_convo)))
                        # Accuracy, Word Class, Model, Entropy, Alpha,
                        Lexi = alpha_to_result_string[PRIOR_TYPE][alpha].split(",")[2]
                        Bert = alpha_to_result_string[PRIOR_TYPE][alpha].split(",")[3]
                        MFS = alpha_to_result_string[PRIOR_TYPE][alpha].split(",")[4]
                        pos_to_prior_to_alpha_epy[POS_SPEC]["results_file"].write("{},{},{},{},{}\n".format(Lexi,POS_SPEC + " " + PRIOR_TYPE, "Lexi Bayesian", entropy_num, alpha))
                        pos_to_prior_to_alpha_epy[POS_SPEC]["results_file"].write("{},{},{},{},{}\n".format(Bert,POS_SPEC + " " + PRIOR_TYPE, "BERT CBOW Bayesian", entropy_num, alpha))
                        pos_to_prior_to_alpha_epy[POS_SPEC]["results_file"].write("{},{},{},{},{}\n".format(Lexi,POS_SPEC + " " + PRIOR_TYPE, "Most Frequent Sense", entropy_num, alpha))

                    except Exception as e:
                        print(e, "could not write to file")
                        continue
logging.info("DONE! Finished classifying all words")
pickle.dump(word_to_embedding, open("word_to_embeddings_n_v.pickle", "wb"))