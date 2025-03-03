string_words = '''United States Declaration of Independence
From Wikipedia, the free encyclopedia
The United States Declaration of Independence is the statement'''

word_list = string_words.split()

word_freq = [word_list.count(n) for n in word_list]
#show string words
print("String:\n {} \n".format(string_words))
#show word in list
print("List:\n {} \n".format(str(word_list)))
#show count of word
print("Pairs (Words and Frequencies:\n {}".format(str(list(zip(word_list, word_freq)))))