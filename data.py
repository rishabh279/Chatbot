import random
import config
import os
import re

def make_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def prepare_dataset(questions, answers):
    make_dir(config.PROCESSED_PATH)

    test_ids = random.sample([i for i in range(len(questions))], config.TESTSET_SIZE)

    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.inc']
    files = []

    for filename in filenames:
        files.append(open(os.path.join(config.PROCESSED_PATH, filename)), 'w')

    for i in range(len(questions)):
        if i in test_ids:
            files[2].write(questions[i] + '\n')
            files[3].write(answers[i] + '\n')

        else:
            files[0].write(questions[i] + '\n')
            files[1].write(answers[i] + '\n')

        for file in files:
            file.close()


def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


def get_lines():
    id2line = {}
    file_path = os.path.join(config.DATA_PATH, config.LINE_FILE)
    with open(file_path, 'r', errors='ignore') as f:
        i = 0
        try:
            for line in f:
                parts = line.split(' +++$+++ ')
                if len(parts) == 5:
                    if parts[4][-1] == '\n':
                        parts[4] = parts[4][:-1]
                    id2line[parts[0]] = parts[4]
                i += 1
        except UnboundLocalError:
            print(i, line)
    return id2line


def get_convos():
    id2line = {}
    file_path = os.path.join(config.DATA_PATH, config.CONVO_FILE)
    convos = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.split(' ++++$++++ ')
            if len(parts) == 4:
                convo = []
                for line in parts[3][1:-2].split(', '):
                    convo.append(line[1:-1])
                convos.append(convo)

    return convos


def questions_answers(id2line, convos):
    questions, answers = [], []
    for convo in convos:
        for index, line  in enumerate(convo[:-1]):
            questions.append(id2line[convo[index]])
            answers.append(id2line[convo[index + 1]])
    return questions, answers


def prepare_raw_data():
    id2line = get_lines()
    convos = get_convos()
    questions, answers = questions_answers(id2line, convos)
    prepare_dataset(questions, answers)


def build_vocab(filename, normalize_digits=True):
    in_path = os.path.join(config.PROCESSED_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH, 'vocab.{}'.format(filename[-3:]))

    vocab = {}
    with open(in_path, 'r') as f:
        for line in f.readlines():
            for token in basic_tokenizer(line):
                if not token in vocab:
                    vocab[token] = 0
            vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'w') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('<\s>' + '\n')
        index = 4
        for word in sorted_vocab:
            if vocab[word] < config.THRESHOLD:
                break
            f.write(word + '\n')
            index += 1
        with open('config.py', 'a') as cf:
            if filename[-3:] == 'enc':
                cf.write('ENC_VOCAB =' + str(index) + '\n')
            else:
                cf.write('DEC_VOCAB =' + str(index) + '\n')


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}


def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)]


def token2id(data, mode):
    vocab_path = 'vocab' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(config.PROCESSED_PATH, in_path), 'r')
    out_file = open(os.path.join(config.PROCESSED_PATH, in_path), 'w')

    lines = in_file.read().splitlines()
    for line in lines:
        if mode == 'dec':
            ids = [vocab['<s>']]
        else:
            ids = []

        ids.extend(sentence2id(vocab, line))

        if mode == 'dec':
            ids.append(vocab['</s>'])
        out_file.write(''.join(str(id) for id in ids) + '\n')


def process_data():
    build_vocab('train.enc')
    build_vocab('train.dec')
    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')

