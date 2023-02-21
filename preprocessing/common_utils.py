import os

def file_to_str(lst):
    """
    Return a string of all the words in the given list of words,
    each separated by a space.

    Args:
        lst (string): filename corresponding to a list of words.
            Assumes each word is separated by a newline character.

    Returns:
        wordlist (string): list of words in the given file
    """
    assert os.path.exists(lst), 'file lst does not exist'

    f = open(lst, encoding="utf8")
    wordlist = f.read().replace('\n', ' ')
    f.close()
    return wordlist


def file_to_list(lst):
    """
    Return a list of all the words in the given list of words.

    Args:
        lst (string): filename corresponding to a list of words.
            Assumes each word is separated by a newline character.

    Returns:
        wordlist (list): list of words in the given file
    """
    assert os.path.exists(lst), 'file lst does not exist'

    f = open(lst, encoding="utf8")
    wordlist = [w.strip('\n') for w in f.readlines()]
    f.close()
    return wordlist
