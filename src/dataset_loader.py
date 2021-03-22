import os
import string
import json
import datasets
import pandas as pd
import re

def clear_punctuation(text):
    return re.sub("'[!@#$]'''``", '', text)


class POSDataset(datasets.GeneratorBasedBuilder):
    """

    """

    decription=""""""
    VERSION = datasets.Version("1.1.0")
    _URL = "data/"

    _URLS = {
        "train": _URL + "train.tsv",
        "dev": _URL + "validate.tsv",
        "test": _URL + "test.tsv"
    }
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="data", version=VERSION, description="This is the whole dataset"),
    ]

    def __description__(self):

        return self.description

    def _info(self):

        return datasets.DatasetInfo(description=self.decription,
                features=datasets.Features(
                    {
                        "id":datasets.Value("string"),
                        "words":datasets.features.Sequence(datasets.Value("string")),
                        "tags":datasets.features.Sequence(datasets.Value("string")),
                        
                        }))
    

    def _split_generators(self, dl_manager):

        urls_to_download = self._URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]})
        ]




    def _generate_examples(
        self, filepath  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        data = open(filepath, "r").readlines()
        for row in data:
            row = row.split("\t")
            words = [x for x in row[1].strip().split(" ")]
            tags = [x for x in row[2].strip().split(" ")]

            assert len(words) == len(tags)
            yield row[0], {
                        "id": row[0].strip(),
                        "words": words,
                        "tags": tags,
                    }
