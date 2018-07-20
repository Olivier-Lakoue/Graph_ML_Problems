from lxml import etree
from glob import glob
import re
import pandas as pd
import numpy as np
import os
import networkx as nx
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Regulations():
    def __init__(self, folder):
        self.reg_files = glob(folder+'*.xml')
        self.filtered_reg_files = self.exclude_SI()

    def exclude_SI(self):
        return [r for r in self.reg_files if re.match(r'^((?!SI).)*$', r)]

    def parse(self):
        titles = []
        files = []
        repealed = []
        references = []

        for i, f in enumerate(self.filtered_reg_files):
            # parse xml file
            doc = etree.parse(f)
            # get the file name
            fname = os.path.basename(f)
            try:
                title = doc.xpath('//LongTitle')[0].text.lower()
            except:
                title = ''
            titles.append(title)
            files.append(fname)
            try:
                repeal = doc.xpath('//Repealed')[0].text
            except:
                repeal = ''
            repealed.append(repeal)

            try:
                xref = doc.xpath('//XRefExternal')
                xref = [x.text.lower() for x in doc.xpath('//XRefExternal') if x.text.lower() != title]
            except:
                xref = []
            references.append(xref)

        self.df = pd.DataFrame({'titles': titles, 'files': files, 'repealed': repealed, 'references': references})
        self.df['class_type'] = [self.title_classification(x) for x in self.df.titles.values]
        return self.df

    def title_classification(self, title):
        """
        Return the identified class of regulation.
        :param title: string
        :return: string
        """
        pattern = re.compile(
            r'(regulation|order|rule|procla|direction|fees|permit|notice|by-law|authorization|list|guideline)')
        if pattern.search(title):
            res = pattern.search(title).group(0)
        else:
            res = 'other'
        return res

class Acts():
    def __init__(self, folder):
        self.reg_files = glob(folder + '*.xml')

    def parse(self):
        titles = []
        short_titles = []
        files = []
        repealed = []
        references = []
        for i, f in enumerate(self.reg_files):
            doc = etree.parse(f)
            filename = os.path.basename(f)
            files.append(filename)
            #     print(filename)
            try:
                short = doc.xpath('//ShortTitle')[0].text.lower()
            except:
                short = ''
            short_titles.append(short)

            try:
                repeal = doc.xpath('//Repealed')[0].text
            except:
                repeal = ''
            repealed.append(repeal)

            try:
                title = doc.xpath('//LongTitle')[0].text.lower()
            except:
                title = ''
            titles.append(title)

            try:
                # get refs and remove self ref
                Xref = [x.text.lower() for x in doc.xpath('//XRefExternal') if x.text.lower() != short]
            except:
                Xref = []
            references.append(Xref)

        self.df = pd.DataFrame({'titles': titles, 'short_title': short_titles, 'files': files, 'repealed': repealed,
                           'references': references})
        return self.df

class CitationsNetwork():
    """
    Build a networkx graph from the 'references' and 'titles' columns of the list of dataframes
    :param list_of_df: list of pandas dataframes
    """
    def __init__(self, list_of_df, max_words=10000, embedding_dim=100, maxlen=100):
        self.max_words = max_words
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.titles_dict = {}
        self.list_df = list_of_df
        self.parse_titles()
        self.parse_graph()
        self.encode_titles()

    def parse_titles(self):
        last_idx = 0
        for df in self.list_df:

            for i, row in df.iterrows():
                if row.titles not in self.titles_dict:
                    self.titles_dict[row.titles] = last_idx
                    last_idx += 1
                for ref in row.references:
                    if ref not in self.titles_dict:
                        self.titles_dict[ref] = last_idx
                        last_idx += 1
    def parse_graph(self):
        nodes_list = [(v, {'title': k}) for k,v in self.titles_dict.items()]
        edges_list = []
        for df in self.list_df:
            for i,row in df.iterrows():
                for ref in row.references:
                    edges_list.append((self.titles_dict[row.titles], self.titles_dict[ref]))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(nodes_list)
        self.graph.add_edges_from(edges_list)

    def encode_titles(self):
        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(list(self.titles_dict.keys()))

        # annotate graph
        for n in nx.nodes(self.graph):
            seq = pad_sequences(tokenizer.texts_to_sequences(([self.graph.node[n]['title']])))
            self.graph.node[n]['encoding'] = np.pad(seq, ((0,0),(0,self.maxlen)), 'constant')[:, :self.maxlen]