import os


def file_to_str(filename):
    """
    Return a string of all the words in the given list of words,
    each separated by a space.

    Args:
        filename (string): filename corresponding to a list of words.
            Assumes each word is separated by a newline character.

    Returns:
        wordlist (string): list of words in the given file
    """
    assert os.path.exists(filename), 'file does not exist'

    f = open(filename)
    wordlist = f.read().replace('\n', ' ')
    f.close()
    return wordlist


def file_to_list(filename):
    """
    Return a list of all the words in the given list of words.

    Args:
        filename (string): filename corresponding to a list of words.
            Assumes each word is separated by a newline character.

    Returns:
        wordlist (list): list of words in the given file
    """
    assert os.path.exists(filename), 'file does not exist'

    f = open(filename)
    wordlist = [w.strip('\n') for w in f.readlines()]
    f.close()
    return wordlist


# TODO: need generic method to fetch publisher/company info from filename. This only works for chinese-games/dataset2
def get_publisher_name(filename):
    """
    Each word list in citizen-lab-data/chinese-games/dataset2 is
    prefaced by the publisher or developer that was used to discover
    the list. Return the name of the publisher given the file name.

    Args:
        filename (string): filename corresponding to a list of words.
            The file must be located in citizen-lab-data/chinese-games/dataset2
            or in citizen-lab-data/chinese-games/dataset2-grouped

    Returns:
        publisher_name (string): name of the publisher
    """
    assert os.path.exists(filename), 'file does not exist'
    assert 'citizen-lab-data/chinese-games/dataset2' in filename, \
        'invalid file name, see docstring for usage'

    file = os.path.basename(filename)
    if '#' in file:
        return file[:file.index('#')]
    return file[:file.index('.')]


def publisher_to_filename(dir):
    """
    Return a mapping of publisher names to all the filenames in the
    directory corresponding to wordlists from that publisher.

    Args:
        dir: directory containing wordlists

    Returns:
        publishers (dict): a mapping from publisher names to a list of
            filenames in the given directory
    """
    # TODO: different logic for open-source directory.
    # currently this is for citizen-lab-data/chinese-games/dataset2 only

    assert os.path.exists(dir), 'directory does not exist'

    publishers = {}
    for file in os.listdir(dir):
        publisher = file[:file.index('#')]
        is_txt = os.path.splitext(file)[1] == '.txt'  # only .txt for now
        if is_txt:
            if publisher in publishers:
                publishers[publisher].append(file)
            else:
                publishers[publisher] = [file]

    return publishers
