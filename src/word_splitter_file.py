
import re
import pprint
import enchant
from ns_log import NsLog


class WordSplitterClass(object):

    def __init__(self):
        self.logger = NsLog("log")
        self.path_data = "../data/"
        self.name_brand_file = "All_Brand.txt"
        self.dictionary_en = enchant.DictWithPWL("en_US", "{0}{1}".format(self.path_data, self.name_brand_file))
        # self.__file_capitalize(self.path_data, self.name_brand_file)

        self.pp = pprint.PrettyPrinter(indent=4)

    def _split(self, gt7_word_list):

        return_word_list = []

        for word in gt7_word_list:
            try:
                ss = {'raw': word,'splitted':[]}

                # If there is a number in the word, it is cleared.
                word = re.sub("\d+", "", word)
                sub_words = []

                if not self.dictionary_en.check(word):
                    #  return this word in the dictionary. If the word is not in the dictionary, proceed to the separation operation.

                    for number in range(len(word), 3, -1): # generation of subwords with length greater than 3
                        for l in range(0, len(word) - number + 1):
                            if self.dictionary_en.check(self.__capitalize(word[l:l + number])):

                                #  When I detect a word, I put * instead of the detected word so that it does not cause fp when detecting other words.
                                w = self.__check_last_char(word[l:l + number])
                                sub_words.append(w)
                                word = word.replace(w, "*" * len(w))

                    rest = max(re.split("\*+", word), key=len)
                    if len(rest) > 3:
                        sub_words.append(rest)

                    split_w = sub_words

                    for l in split_w:
                        for w in reversed(split_w):

                            """
                            If a detected word is also in a larger word, it is fp.
                            I cleaned these. For example, secure, cure.
                            The word cure is removed from the list.
                            """

                            if l != w:  # todo edit distance eklenecek
                                if l.find(w) != -1 or l.find(w.lower()) != -1:
                                    sub_words.remove(w)

                    if len(sub_words) == 0:
                        #  eğer hiç kelime bulunamadıysa ham kelime olduğu gibi geri döndürülür.
                        sub_words.append(word.lower())
                else:
                    sub_words.append(word.lower())

                ss['splitted']=sub_words
                return_word_list.append(ss)
            except:
                self.logger.debug("|"+word+"| işlenirken hata")
                # self.logger.error("word_splitter.split()  /  Error : {0}".format(format_exc()))

        return return_word_list

    def _splitl(self, gt7_word_list):

        result = []

        for val in self._split(gt7_word_list):
            result += val["splitted"]

        return result

    def _splitw(self, word):

        word_l = []
        word_l.append(word)

        result = self._split(word_l)

        return result

    def __check_last_char(self, word):

        confusing_char = ['s', 'y']
        last_char = word[len(word)-1]
        word_except_last_char = word[0:len(word)-1]
        if last_char in confusing_char:
            if self.dictionary_en.check(word_except_last_char):
                return word_except_last_char

        return word

    def __clear_fp(self, sub_words):

        length_check = 0
        for w in sub_words:
            if (length_check + len(w)) < self.length+1:
                length_check = length_check + len(w)
            else:
                sub_words.remove(w)

        sub_words = self.__to_lower(sub_words)
        return sub_words

    def __to_lower(self, sub_words):

        lower_sub_list = []

        for w in sub_words:
            lower_sub_list.append(str(w.lower()))

        return lower_sub_list

    def __capitalize(self, word):
        return word[0].upper() + word[1:]

    def __file_capitalize(self, path, name):

        """
        In order for special words to be checked in the enchant package, their first letter must be capitalized.
        Before checking a word, I capitalize the first letter and then ask the dictionary.
        For this reason, I capitalized the first letters of the words in the file and saved them and used them that way.
        :return: 
        """

        personel_dict_txt = open("{0}{1}".format(path, name), "r")

        personel_dict = []

        for word in personel_dict_txt:
            personel_dict.append(self.__capitalize(word.strip()))

        personel_dict_txt.close()

        personel_dict_txt = open("{0}{1}-2".format(path, name), "w")

        for word in personel_dict:
            personel_dict_txt.write(word+"\n")
