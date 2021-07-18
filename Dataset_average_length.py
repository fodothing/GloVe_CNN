# 平均每条评论含有的单词个数

def count_word_sentence(comments):
    # 求平均每条评论含有多少单词
    all_words = 0
    all_sentences = len(comments)
    for i in range(len(comments)):
        all_words += len(comments[i])

    return all_sentences, all_words

