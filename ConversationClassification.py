class ConversationClassification:
    def __init__(self, classified_sense, syn_context,
                 cos_similarity,
                 target_context, childes_pos, target_word, conversation,
                 wn_freq_pos, main_sentence):
        self.classified_sense = classified_sense
        self.bert_sense = None
        self.syn_context = syn_context
        self.cos_similarity = cos_similarity
        self.target_context = target_context
        self.childes_pos = childes_pos
        self.target_word = target_word
        self.conversation = conversation
        self.wn_freq_pos = wn_freq_pos
        self.gt_sense_list = []
        self.gt_str_sense_list = []
        self.main_sentence = main_sentence
        self.file = ""
        self.window_size = None # default
        self.main_sent_index = None
        self.id = ""
        self.sense_to_votes_sorted = []
        self.sense_to_conv_lists = []
        self.all_votes = 0
        self.top_uniform_tags = []

    def __str__(self):
         string = "==============================\n"
         string += ("==== {}, {} , {} ====:\n".format(self.target_word, self.gt_str_sense_list, "MFQ" if self.wn_freq_pos.name() in self.gt_str_sense_list else "POLY" ))
         string += ("==== Target Sentence {} ====:\n {}\n".format(self.id, " ".join(["" if w[0] == "CLITIC" else w[0] for w in self.main_sentence])))
         string+= ("==== Conversation ({}) ====:\n {}\n".format(self.target_word, self.conversation ))
         string += ("==== Lexical Chaining Bayesian sense ==== \nName: {}\nDefinition: {}\nExamples:"
                    " {}\n".format( self.classified_sense.name(), self.classified_sense.definition(), self.classified_sense.examples()))
         string += ("=== Ground Truth (Berkely) Sense === \n")
         for gt_sense in self.gt_sense_list:
            string += "{} : {}\n".format(gt_sense.name(), gt_sense.definition())
         string += ("=== WordNet Most Frequent Sense === \n{} : {}\n".format(self.wn_freq_pos.name(), self.wn_freq_pos.definition()))
         string += ("=== Bert Sense === \n{} : {}\n".format(self.bert_sense.name(), self.bert_sense.definition()))

         string += ("==== POS Tag: {}  Window Size: {} From file: {}"
                    " Line: {} ==== \n".format(self.childes_pos, self.window_size,
                                               self.file, self.main_sent_index,
                                               ))
         string += ("==== COS Symilarity {} ==== \n".format(self.cos_similarity))
         string += ("==== Conversation Contexts ==== \n {}\n".format(self.target_context))
         string += ("==== Sense Contexts ==== \n {}\n".format(self.syn_context))
         string += "=== Top Uniformly Distributed Tags === \n"
         for sense in self.top_uniform_tags:
             string += "{},".format(sense)
         string += "\n=== Sorted Voting Order === \n"
         for sense, sum in self.sense_to_votes_sorted:
             string += "{} : {} / {} = {}, sum cos_similarity = {}\n".format(
                     sense.name(), len(self.sense_to_conv_lists[sense]), self.all_votes,
                    len(self.sense_to_conv_lists[sense])/self.all_votes, sum)
         string += "==============================\n"
         return string
