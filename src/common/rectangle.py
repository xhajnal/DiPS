class My_Rectangle:
    def __init__(self, region, is_white=False, model=(None, None)):
        self.region = region
        self.is_white = is_white
        self.model = model

    def is_white(self):
        return self.is_white

    def get_model(self):
        return self.model
