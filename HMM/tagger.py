import numpy as np
import myHMM

def row_normalization(X):
    """
    按行归一化
    :param X: 矩阵
    :return: 归一化后的矩阵
    """
    X.dtype = 'float'
    for l in range(X.shape[0]):
        sum = np.sum(X[l])
        X[l] = X[l]/sum
    return X


def load_file(file_name, charset='utf-8'):
    """
    读取文件，按列返回列表
    :param file_name: 文件路径
    :return: 文本内容列表
    """
    line_list = []
    with open(file_name, encoding=charset) as f:
        for line in f.readlines():
            line = line.strip()
            if ']' in line:
                line.replace(']', '  ')
            if '[' in line:
                line.replace('[', '  ')
            line = line.strip('\n')
            if len(line) >0:
                line_list.append(line)
    return line_list

def loadMetrix():
    """
    load metrix
    :return:转移矩阵A,发射矩阵B,初始分布pi
    """
    lines = load_file('./data/19980206.txt')
    vocab_set = set()
    cixin_set = set()
    for line in lines:
        words = line.strip().split()
        for word in words:
            tmp = word.split('/')
            if len(tmp) < 2:continue
            if '/' in word and tmp[0] not in vocab_set:
                vocab_set.add(tmp[0])
            if '/' in word and tmp[1] not in cixin_set:
                cixin_set.add(tmp[1])
    cixin_num = len(cixin_set)
    cixin_list = []
    vocab_list = []
    for i in vocab_set:
        vocab_list.append(i)
    for i in cixin_set:
        cixin_list.append(i)
    cixin_map = dict(zip(cixin_list, range(cixin_num)))  # cixin_map['j']表示该词性在词性表对应的索引位置

    trans_A = np.zeros((cixin_num, cixin_num))  # 转移矩阵
    pi = np.zeros(cixin_num, dtype=int)

    for line in lines:
        word = line.strip().split()
        if len(word) >= 2:
            word = word[1]
            if '/' in word:
                cixin = (word.split('/')[1])
                pi[cixin_map[cixin]] += 1

    pi = pi  / (np.sum(pi))
    #转移矩阵A
    pre_cixin = ''
    cixin = ''
    for line in lines:
        words = line.strip().split()
        for word in words:
            if '/' in word:
                cixin = (word.split('/')[1])
                try:
                    trans_A[cixin_map[pre_cixin]][cixin_map[cixin]] += 1
                except KeyError:
                    pass
                pre_cixin = cixin


    trans_A = row_normalization(trans_A)
    vocab_map = dict(zip(vocab_list, range(len(vocab_list))))
    emitter_B = np.zeros((cixin_num, len(vocab_list)))
    for line in lines:
        words = line.strip().split()
        for word in words:
            if '/' in word:
                tmp = word.split('/')
                vocab = tmp[0]
                cixin = tmp[1]
                    #print(word)
                try:
                    emitter_B[cixin_map[cixin]][vocab_map[vocab]] += 1
                except KeyError:
                    # print vocab, '不在词库内 忽略不计'
                    pass

    emitter_B = row_normalization(emitter_B)

    return trans_A, emitter_B, pi, vocab_map, cixin_map
if __name__ == '__main__':
    hmm = myHMM.myHMM()
    a, b, pi, vocab_map, cixin_map= loadMetrix()

    index_cixin = dict()
    for key in cixin_map.keys():
        index_cixin[cixin_map[key]] = key
    result = 0      #总词性数
    hmm_result = 0  #预测正确的词性数
    sentences = load_file('./data/199801.txt')
    errors = []
    for sentence in sentences:
        sentence = sentence.strip().strip('/n')
        if len(sentence) <= 0:
            continue

        sentence = sentence.split()[1:]
        words = []
        cixins = []
        o = []
        for word in sentence:
            if '/' in word:
                tmp = word.split('/')
                words.append(tmp[0])
                cixins.append(tmp[1])
        for key in words:
            if key in vocab_map:
                o.append(vocab_map[key])
            else:
                o.append(0)
        result = result+len(cixins)
        flag =False
        hmm_cixins = []
        if len(o) >0:
            path = hmm.HMMViterbi(a, b, o, pi)

            for index in path:
                hmm_cixins.append(index_cixin[int(index)])
            if len(cixins)>len(hmm_cixins):
                length = len(hmm_cixins)
            else:
                length = len(cixins)
            for i in range(length):
                if cixins[i] == hmm_cixins[i]:
                    hmm_result = hmm_result+1
        #         elif not flag :
        #             flag = True
        # if flag :
        #     errors.append(words+hmm_cixins)

    # with open('tmp.txt', 'w+') as f:
    #     for error in errors:
    #         f.write(str(error)+'/n')
    #
    print(hmm_result, result, hmm_result/result)



    #手动输入预测部分
    # sentence = ' '
    #
    # while sentence:
    #     sentence = input('请输入分好词的句子:')#向  广大  职工  祝贺  新年  ，  对  节日  坚守  岗位  的  同志  们  表示  慰问
    #     list  = sentence.strip().split()
    #     o = []
    #     #print(pi)
    #     for key in list:
    #         if key in vocab_map:
    #             o.append(vocab_map[key])
    #         else:
    #             o.append(0)
    #     path = hmm.HMMViterbi(a, b, o, pi)
    #     cixin = []
    #     for index in path:
    #         cixin.append(index_cixin[int(index)])
    #
    #     # for value in path:
    #     #     for key in cixin_map.keys():
    #     #         if cixin_map[key] == value:
    #     #             cixin.append(key)
    #     tag = []
    #     for i  in range(len(list)):
    #         tag.append(list[i]+'/'+cixin[i])
    #
    #     print(tag)

    