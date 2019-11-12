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
