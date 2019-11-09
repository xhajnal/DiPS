import textwrap


## source:  https://stackoverflow.com/questions/1166317/python-textwrap-library-how-to-preserve-line-breaks
class DocumentWrapper(textwrap.TextWrapper):

    def wrap(self, text):
        """ Returns the text with lines split if longer than given size """
        split_text = text.split('\n')
        lines = [line for para in split_text for line in textwrap.TextWrapper.wrap(self, para)]
        return lines
