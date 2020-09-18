import os
import pickle


def write_to_file(output_file_path, output, silent, append=False):
    """  Generic writing to a file

    Args:
        output_file_path (string): path of the output file
        output (string): text to be written into the file
        silent (bool): if silent command line output is set to minimum
        append (bool): if True appending instead of writing from the start
    """
    if not silent:
        print("output here: " + str(output_file_path))
    if append:
        with open(output_file_path, 'a') as output_file:
            output_file.write(output)
    else:
        with open(output_file_path, 'w') as output_file:
            output_file.write(output)
            output_file.close()


def pickle_load(file):
    """ Returns loaded pickled data

    Args:
        file (string or Path): filename/filepath
    """

    filename, file_extension = os.path.splitext(file)

    if file_extension == ".p":
        with open(file, "rb") as f:
            return pickle.load(f)
    elif file_extension == "":
        with open(os.path.join(file, ".p"), "rb") as f:
            return pickle.load(f)
    else:
        raise Exception("File extension does not match", f"{file} does not seem to be pickle file!")


def pickle_dump(what, file):
    """ Dumps given file as pickle

    Args:
        what (object): something to be pickled
        file (string or Path): filename/filepath
    """
    with open(file, 'wb') as f:
        pickle.dump(what, f)
